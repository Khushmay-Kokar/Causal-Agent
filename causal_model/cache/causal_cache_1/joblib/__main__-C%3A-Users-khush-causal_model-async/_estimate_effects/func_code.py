# first line: 195
@mem.cache(ignore=["df"])
async def _estimate_effects(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    controls: list,
    counterfactual_value: Optional[float] = None,
    interpret: bool = False,
    df_hash: Optional[str] = None
) -> Dict[str, any]:

    def get_rules(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
            for i in tree_.feature
        ]
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], conditions + [f"{name} ≤ {threshold:.2f}"])
                recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.2f}"])
            else:
                value = tree_.value[node][0][0]
                rule = " and ".join(conditions) if conditions else "Always"
                rules.append((rule, int(round(value))))
        recurse(0, [])
        return rules

    def explain_policy_rules(tree_text):
        explanation = []
        current_conditions = []

        for line in tree_text.split("\n"):
            depth = line.count("|   ")
            line_clean = line.replace("|   ", "").strip()

            # Truncate to current depth
            current_conditions = current_conditions[:depth]

            if "class:" in line_clean:
                treatment = int(line_clean.split(":")[-1].strip())
                condition_str = " and ".join(current_conditions) if current_conditions else "Always"
                explanation.append(f"If {condition_str}, then the recommended policy is: Treatment {treatment}.")
            elif "<=" in line_clean or ">" in line_clean:
                current_conditions.append(line_clean)

        return explanation

    # --- 1. Fit estimator ---
    X = df[controls]
    T = df[treatment].values.reshape(-1, 1)
    Y = df[outcome]
    est = _pick_estimator(Y, T, X)
    await asyncio.to_thread(est.fit, Y, T, X=X) # Use asyncio to run fit in a separate thread

    # --- 2. Basic effects ---
    ate = est.ate(X)
    cate = est.effect(X)
    cate_summary = _summarize_cate(cate)
    mid = len(Y) // 2
    ite_sample = float(await asyncio.to_thread(est.effect, X.iloc[[mid]]))

    X_cf = X.copy()
    T_cf = np.full_like(T, counterfactual_value if counterfactual_value is not None else T.mean())
    cf_effect = await asyncio.to_thread(est.effect, X_cf, T0=T, T1=T_cf)

    # --- 3. Root causes ---
    if hasattr(est, "feature_importances_"):
        fi = est.feature_importances_
        root_causes = sorted(zip(controls, fi), key=lambda x: -x[1])[:5]
    else:
        root_causes = []

    feature_differences = []
    for feat, _ in root_causes:
        low_group = X[X[feat] < X[feat].median()]
        high_group = X[X[feat] >= X[feat].median()]
        if len(low_group) == 0 or len(high_group) == 0:
            continue
        eff_low = await asyncio.to_thread(est.effect, low_group)
        eff_high = await asyncio.to_thread(est.effect, high_group)
    #     feature_differences.append({
    #         "feature": feat,
    #         "low_effect": float(eff_low),
    #         "high_effect": float(eff_high),
    #         "effect_diff": float(eff_high - eff_low)
    #     })

    # root_summary = "\n".join([
    #     f"When '{f['feature']}' is high, the treatment effect {'increases' if f['effect_diff'] > 0 else 'decreases'} by {abs(f['effect_diff']):.3f} units."
    #     for f in feature_differences
    # ])
    # ✅ Ensure scalar conversion
        eff_low_val = float(eff_low.mean())  # use .mean() or .item() if single value
        eff_high_val = float(eff_high.mean())

        feature_differences.append({
            "feature": feat,
            "low_effect": eff_low_val,
            "high_effect": eff_high_val,
            "effect_diff": eff_high_val - eff_low_val
        })

    root_summary = "\n".join([
        f"When '{f['feature']}' is high, the treatment effect {'increases' if f['effect_diff'] > 0 else 'decreases'} by {abs(f['effect_diff']):.3f} units."
        for f in feature_differences
    ])
    # --- 4. Confidence Interval ---
    if hasattr(est, "cate_interval"):
        lb, ub = await asyncio.to_thread(est.effect_interval, X)
        ci_summary = {"lower_bound_mean": float(lb.mean()), "upper_bound_mean": float(ub.mean())}
    else:
        ci_summary = {}

    result = {
        "ate": float(ate),
        "cate_summary": cate_summary,
        "ite_sample": ite_sample,
        "counterfactual_example": float(cf_effect.mean()),
        "root_causes": root_causes,
        "uncertainty": ci_summary,
        "root_cause_differences": feature_differences,
        "root_cause_summary_text": root_summary
    }
    # --- 5. CATE and Policy Trees Visuals ---
    if interpret:
        try:
            X.columns = X.columns.map(str)

            # CATE Tree
            cate_interpreter = SingleTreeCateInterpreter(max_depth=3)
            await asyncio.to_thread(cate_interpreter.interpret, est, X)
            fig, ax = await asyncio.to_thread(plt.subplots, figsize=(10, 6))
            await asyncio.to_thread(cate_interpreter.plot, ax=ax, feature_names=controls)
            buf = BytesIO()
            await asyncio.to_thread(fig.savefig, buf, format="png")
            result["cate_interpreter_tree_png"] = base64.b64encode(buf.getvalue()).decode()
            result["cate_interpreter_rules_llm"] = [
                f"If {cond}, then estimated treatment effect (CATE) is {val:.2f}"
                for cond, val in get_rules(cate_interpreter.tree_model_, list(X.columns))
            ]
            await asyncio.to_thread(plt.close, fig)

            # Policy Tree
            if np.unique(T).shape[0] == 2:
                policy_interpreter = SingleTreePolicyInterpreter(max_depth=3)
                await asyncio.to_thread(policy_interpreter.interpret, est, X=X)
                fig, ax = await asyncio.to_thread(plt.subplots, figsize=(10, 6))
                await asyncio.to_thread(policy_interpreter.plot, ax=ax, feature_names=controls)
                buf2 = BytesIO()
                await asyncio.to_thread(fig.savefig, buf2, format="png")
                result["policy_tree"] = base64.b64encode(buf2.getvalue()).decode()
                tree_text = export_text(policy_interpreter.tree_model_, feature_names=list(X.columns))
                result["policy_tree_text"] = tree_text
                result["policy_tree_rules_llm"] = explain_policy_rules(tree_text)
                await asyncio.to_thread(plt.close, fig)
        except Exception as e:
            result["cate_interpreter_tree_png"] = f"Error generating tree: {str(e)}"
            result["cate_interpreter_rules_llm"] = f"Error: {str(e)}"
            result["policy_tree"] = f"Error generating policy tree: {str(e)}"
            result["policy_tree_rules_llm"] = f"Error: {str(e)}"

    return result
