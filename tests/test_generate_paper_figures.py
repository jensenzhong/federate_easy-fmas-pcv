import inspect
import unittest


class GeneratePaperFiguresTests(unittest.TestCase):
    def test_new_mainline_figure_constants_match_closure_plan(self):
        import scripts.generate_paper_figures as figures

        self.assertEqual(
            figures.MAINLINE_SCENARIOS,
            ["A", "A_prime", "B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"],
        )
        self.assertEqual(
            figures.FEDERATED_MAINLINE_SCENARIOS,
            ["B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"],
        )
        self.assertEqual(
            figures.SCATTER_SCENARIOS,
            ["A_prime", "B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"],
        )
        self.assertEqual(
            figures.BIAS_CORRECTION_SCENARIOS,
            ["FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"],
        )

    def test_visible_labels_do_not_use_abc_scenario_names(self):
        import scripts.generate_paper_figures as figures

        visible_source = "\n".join(
            inspect.getsource(func)
            for func in [
                figures.fig1_scenario_comparison,
                figures.fig2_convergence_comparison,
                figures.fig3_llm_strategy_timeline,
                figures.fig4_client_mape_trends,
                figures.fig5_prediction_scatter,
                figures.fig9_bias_correction,
            ]
        )

        forbidden = [
            "A (GBR)",
            "A' (NN)",
            "B (FedAvg)",
            "C (MAS-FL",
            "Scenario C",
            "B vs C",
        ]
        for label in forbidden:
            self.assertNotIn(label, visible_source)

    def test_llm_timeline_uses_candidate_selection_and_gate_fields(self):
        import scripts.generate_paper_figures as figures

        source = inspect.getsource(figures.fig3_llm_strategy_timeline)

        for expected in [
            "mas_vg_fedyogi_tr_round_metrics.csv",
            "mas_vg_fedyogi_tr_llm_decisions.jsonl",
            "requested_candidate_id",
            "selected_candidate_id",
            "gate_status",
        ]:
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

        for legacy in [
            "scene_C_llm_decisions.jsonl",
            "chosen_strategy_name",
            "lr_scale",
            "epoch_delta",
        ]:
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, source)


if __name__ == "__main__":
    unittest.main()
