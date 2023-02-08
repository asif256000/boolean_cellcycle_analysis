import time
from pathlib import Path

import pandas as pd

from state_calc import CellCycleStateCalculation
from utils import all_perturbation_generator, generate_categorical_hist, generate_histogram

if __name__ == "__main__":
    organism = "mammal"  # Options: ["yeast", "mammal"]
    if organism.lower() == "yeast":
        from yeast_inputs import cyclins, original_graph
    else:
        from mammal_inputs import cyclins, original_graph

    test_state = CellCycleStateCalculation(cyclins=cyclins, organism=organism, detailed_logs=False)

    test_state.set_custom_connected_graph(graph=original_graph, graph_identifier="Original Graph")
    graph_score, g1_graph_score, final_state_counts = test_state.generate_graph_score_and_final_states(
        view_state_table=False, view_final_state_count_table=False
    )
    score_df_list = [["Original Graph", graph_score, g1_graph_score, graph_score + g1_graph_score]]

    og_g_score, og_g1_score = test_state.get_optimal_scores()
    graph_score_list, g1_graph_score_list, sum_graph_score_list, perturb_freq = list(), list(), list(), dict()

    t1 = time.time()
    i = 0
    for modified_matrix, matrix_modifier in all_perturbation_generator(nodes=cyclins, graph=original_graph):
        test_state.set_custom_connected_graph(graph=modified_matrix, graph_identifier=matrix_modifier)
        graph_score, g1_graph_score, final_state_counts = test_state.generate_graph_score_and_final_states(
            view_state_table=False, view_final_state_count_table=False
        )
        # (
        #     graph_score,
        #     g1_graph_score,
        #     final_state_counts,
        #     matrix_modifier,
        # ) = test_state.mp_generate_graph_score_and_final_states(graph_info=(modified_matrix, matrix_modifier))
        graph_param = (graph_score - og_g_score) / og_g_score
        graph_score_list.append(graph_param)
        g1_graph_param = "NA"
        sum_graph_score_param = "NA"
        if g1_graph_score != 0:
            if g1_graph_score <= og_g1_score:
                for perturb in matrix_modifier.split(" | "):
                    if perturb not in perturb_freq.keys():
                        perturb_freq[perturb] = 1
                    else:
                        perturb_freq[perturb] += 1
            g1_graph_param = (g1_graph_score - og_g1_score) / og_g1_score
            og_sum_score = og_g_score + og_g1_score
            sum_graph_score_param = ((graph_score + g1_graph_score) - og_sum_score) / og_sum_score
            g1_graph_score_list.append(g1_graph_param)
            sum_graph_score_list.append(sum_graph_score_param)
        score_df_list.append([matrix_modifier, graph_score, g1_graph_score, graph_score + g1_graph_score])
        if i > 300:
            break
        i += 1

    print(f"Time taken: {time.time() - t1} seconds...")

    score_df = pd.DataFrame(score_df_list, columns=["Graph Mod Id", "Graph Score", "G1 Graph Score", "Sum Graph Score"])
    score_df.to_csv("Scores.csv")

    print(f"{perturb_freq=}")

    fig_folder = Path(f"figures/{time.strftime('%m%d_%H%M%S', time.gmtime(time.time()))}")
    fig_folder.mkdir(parents=True, exist_ok=True)

    generate_histogram(graph_score_list, img_filename="GraphScore", plot_title="Graph Score Frequency")
    Path("GraphScore.png").rename(fig_folder / "GraphScore.png")

    generate_histogram(g1_graph_score_list, img_filename="G1GraphScore", plot_title="G1 Graph Score Frequency")
    Path("G1GraphScore.png").rename(fig_folder / "G1GraphScore.png")

    generate_histogram(sum_graph_score_list, img_filename="SumGraphScore", plot_title="Sum Graph Score Frequency")
    Path("SumGraphScore.png").rename(fig_folder / "SumGraphScore.png")

    generate_categorical_hist(
        dict(sorted(perturb_freq.items(), key=lambda x: x[1], reverse=True)),
        img_filename="PerturbationFrequency",
        plot_title="Optimal Perturbation Frequency",
    )
    Path("PerturbationFrequency.png").rename(fig_folder / "PerturbationFrequency.png")
