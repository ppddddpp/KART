from experiment_case import (
    case1_fmnist,
    case2_cifar100,
    case3_cifar100_rank,
    case4_synthetic_xai,
    case5_q_d_grid,
    case6_ablation,
    case7_gradient_theory,
    case8_d_m_grid,
    case9_vit_backbone,
    case10_vit_kart,
    case11_vit_final_tuning,
    case12_equalized_xai
)

if __name__ == '__main__':
    dataset_seed = 42
    seeds_list = [1, 2, 3]

    case1_fmnist(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case2_cifar100(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case3_cifar100_rank(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case4_synthetic_xai(dataset_seed=dataset_seed)
    case5_q_d_grid(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case6_ablation(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case7_gradient_theory()
    case8_d_m_grid(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case9_vit_backbone(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case10_vit_kart(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case11_vit_final_tuning(dataset_seed=dataset_seed, seeds_list=seeds_list)
    case12_equalized_xai(dataset_seed=dataset_seed)