 using Statistics
using NPZ
using CSV, DataFrames, PrettyTables
using Random
cd(@__DIR__)



function collect_stat_difference_for_seeds(baseline, model, stat)
    baseline_per_seed = length(keys(baseline)) > 1 ?
                        Dict(config => baseline[parse(Int, match(r"_S(\d+)", config).captures[1])][stat] for config in keys(model)) :
                        Dict(config => baseline[first(keys(baseline))][stat] for config in keys(model))


    diffs = [vec((model[config][stat] .- baseline_per_seed[config]))
             for config in keys(model)]

    return diffs
end

function blocked_bootstrap(x; blocksize=round(Int, length(x)^(1 / 3)), agg::Function=mean, rng=MersenneTwister(42), resamples=10000)
    n = length(x)
    blocksize = min(blocksize, n)
    numblocks = n - blocksize + 1

    return [agg(vcat([x[i:i+(blocksize-1)] for i in rand(rng, 1:numblocks, ceil(Int, n / blocksize))]...)[1:n]) for _ in 1:resamples]
end

function blocked_bootstrap_seeds(arrays::Vararg{AbstractVector}; blocksize=round(Int, length(arrays[1])^(1 / 3)), agg::Function=mean, resamples=10000)
    all(length(arrays[1]) == length(a) for a in arrays) || throw(ArgumentError("All arrays must be the same length"))

    rngs = [MersenneTwister(42) for _ in 1:length(arrays)]

    return agg(hcat([blocked_bootstrap(array; blocksize=blocksize, agg=agg, rng=rngs[i], resamples=resamples) for (i, array) in enumerate(arrays)]...), dims=2)
end

function get_confidence_interval(x; level=0.95, correction_n=1)
    alpha = 1 - level
    alpha = alpha / correction_n
    ci_lower, ci_upper = quantile(vec(x), [alpha / 2, 1 - alpha / 2])

    println("$(Int(level*100))% confidence interval: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]")

    return (ci_lower, ci_upper)
end


function calculate_metrics(y_pred, y_true)
    residuals = y_pred .- y_true
    pred_signs = sign.(diff(y_pred; dims=2))
    true_signs = sign.(diff(y_true; dims=2))
    return (
        mae_complete=(mean(abs.(residuals))),
        mae_elementwise=(mean(abs.(residuals); dims=2)),
        rmse_complete=(sqrt(mean(residuals .^ 2))),
        rmse_elementwise=(sqrt.(mean(residuals .^ 2; dims=2))),
        complement_pta=(1 .- mean(pred_signs .== true_signs)),
        complement_pta_elementwise=(1 .- mean(pred_signs .== true_signs; dims=2))
    )
end

function transform_ODE_output(ode_output; h=7)
    ode_output_transformed = [ode_output[i+j-1] for i in 1:(length(ode_output)-h+1), j in 1:h]
    return (reshape(ode_output_transformed, size(ode_output_transformed, 1), size(ode_output_transformed, 2), 1))
end

function get_metrics_for_ode(dataset, filename, ode_cutoff, ode_column, horizon)
    results = Dict()
    values = values = CSV.read("$(dataset)/ODE/$(filename)_ode_forecast.csv", DataFrame)[ode_cutoff:end, ode_column]
    if dataset == "covid"
        values = diff(values)
    elseif occursin("rotifers", dataset)
        values = values ./ 10^3
    end
        
    y_pred = transform_ODE_output(values, h=horizon)
    y_true = npzread("$dataset/ODE/y_true.npy")

    results[0] = calculate_metrics(y_pred, y_true)

    return (results)
end

function get_metrics_for_all_seeds(dataset, model_type, best_model_name, configs)
    results = Dict()
    for config in configs

        y_pred = npzread("$dataset/$model_type/$config/$best_model_name-y_pred.npy")
        y_true = npzread("$dataset/$model_type/$config/y_true.npy")

        results[config] = calculate_metrics(y_pred, y_true)

    end
    return (results)
end

function get_summary_over_seeds(metrics, non_elementwise_metrics)


    summary_stats = Dict()
    for metric in non_elementwise_metrics
        metric_values = [v[metric] for v in values(metrics)]
        summary_stats[metric] = (
            mean=mean(metric_values),
            median=median(metric_values),
            std=std(metric_values)
        )
    end
    return (summary_stats)
end

function compare_models(baseline, model; stats=[:mean, :median])
    diffs = Dict()
    for metric in keys(baseline)
        baseline_i = baseline[metric]
        model_i = model[metric]

        stat_diffs = Dict()
        for stat in stats
            stat_diffs[stat] = (
                abs_diff=(model_i[stat] - baseline_i[stat]),
                rel_diff=((model_i[stat] - baseline_i[stat]) / baseline_i[stat])
            )
        end

        diffs[metric] = stat_diffs
    end

    return (diffs)
end

function pretty_print_metrics(d::Dict; indent=0)
    prefix = " "^indent
    for (k, v) in d
        if v isa Dict
            println("$prefix$k:")
            pretty_print_metrics(v; indent=indent + 4)
        elseif v isa NamedTuple || v isa Tuple
            # handle NamedTuple or plain Tuple
            vals = join(["$(name)=$(val)" for (name, val) in pairs(v)], ", ")
            println("$prefix$k: ($vals)")
        else
            println("$prefix$k = $v")
        end
    end
end

non_elementwise_metrics = [:mae_complete, :rmse_complete, :complement_pta]
elementwise_metrics = [:mae_elementwise, :rmse_elementwise, :complement_pta_elementwise]

size_label(n::AbstractString) = Dict(
    "0" => "S",
    "1" => "M",
    "2" => "L",
    "3" => "XL"
)[n]

dataset_label(n::AbstractString) = Dict(
    "covid" => "COV-19",
    "algae-rotifers-coherent" => "R-A (coh.)",
    "algae-rotifers-incoherent" => "R-A (inc.)",
    "lynx-hares" => "L-H"
)[n]

rows_table2 = DataFrame()
rows_CI = DataFrame()
for dataset in [
    Dict("name" => "covid", "ts" => "1000", "ic" => "3", "kp" => "3", "model" => "PyTorch_Lightning_LSTM_Neural_Network", "ode_cutoff" => 91, "ode_column" => 1, "horizon" => 7),
    Dict("name" => "algae-rotifers-coherent", "ts" => "100", "ic" => "1", "kp" => "3", "model" => "PyTorch_Lightning_LSTM_Neural_Network", "ode_cutoff" => 8, "ode_column" => 2, "horizon" => 3),
    Dict("name" => "algae-rotifers-incoherent", "ts" => "1", "ic" => "0", "kp" => "0", "model" => "PyTorch_Lightning_LSTM_Neural_Network", "ode_cutoff" => 8, "ode_column" => 2, "horizon" => 3),
    Dict("name" => "lynx-hares", "ts" => "1000", "ic" => "1", "kp" => "0", "model" => "PyTorch_Lightning_Convolutional_Neural_Network", "ode_cutoff" => 20, "ode_column" => 2, "horizon" => 5) 
    ]
    dataset_name = dataset["name"]
    ts = dataset["ts"]
    ic = dataset["ic"]
    kp = dataset["kp"]
    model_name = dataset["model"]
    print("\n\n\nDATASET: $dataset_name \n\n")

    configs = ["TS$(ts)_IC$(ic)_P$(kp)" * "_S$seed" for seed in [10 17 42 93 97]]

    print(configs)

    DL = get_metrics_for_all_seeds(dataset_name, "DL", dataset["model"], [10 17 42 93 97])
    TL = get_metrics_for_all_seeds(dataset_name, "TL", dataset["model"], configs)

    

    DL_stats = get_summary_over_seeds(DL, non_elementwise_metrics)
    # pretty_print_metrics(DL_stats)
    TL_stats = get_summary_over_seeds(TL, non_elementwise_metrics)
    # pretty_print_metrics(TL_stats)

    print("\n\n")
    print("DL baseline\n")
    pretty_print_metrics(DL_stats)
    print("\n\n")
    print("TL\n")
    pretty_print_metrics(TL_stats)
    print("\n\n")

    ODE = get_metrics_for_ode(dataset_name, "$dataset_name", dataset["ode_cutoff"], dataset["ode_column"], dataset["horizon"])
    # pretty_print_metrics(ODE)
    ODE_stats = get_summary_over_seeds(ODE, non_elementwise_metrics)
    print("ODE baseline\n")
    pretty_print_metrics(ODE_stats)
    print("\n\n")
    # pretty_print_metrics(ODE_stats)
    # TODO only for algae-rotifers-incoherent the ODE metrics don't exactly match the ones calculated by Julian


    print("DEEP LEARNING BASELINE \n\n")
    dl_vs_tl = compare_models(DL_stats, TL_stats; stats=[:mean])
    pretty_print_metrics(dl_vs_tl)
    print("MAE ")
    ci_mae_dl_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(DL, TL, :mae_elementwise)...); correction_n=4)
    
    print("1-PTA ")
    ci_pta_dl_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(DL, TL, :complement_pta_elementwise)...); correction_n=4)

    print("RMSE ")
    ci_rmse_dl_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(DL, TL, :rmse_elementwise)...); correction_n=4)
    
    print("\n\n")


    print("ODE BASELINE \n\n")
    ode_vs_tl = compare_models(ODE_stats, TL_stats; stats=[:mean])
    pretty_print_metrics(ode_vs_tl)
    print("MAE ")
    ci_mae_ode_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(ODE, TL, :mae_elementwise)...); correction_n=4)
    print("1-PTA ")
    ci_pta_ode_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(ODE, TL, :complement_pta_elementwise)...); correction_n=4)
    print("RMSE ")
    ci_rmse_ode_tl = get_confidence_interval(blocked_bootstrap_seeds(collect_stat_difference_for_seeds(ODE, TL, :rmse_elementwise)...); correction_n=4)
    print("\n\n\n")

    print("------------------------------------------------------")

    create_output_string(metric) = string(round(metric, sigdigits=2), "%")
    create_ci_string(conf_int) = "[" * join(round.(conf_int, digits=2), ", ") * "]"
    get_model_string(model_name) = occursin("Conv", model_name) ? "CNN" : occursin("LSTM", model_name) ? "LSTM" : nothing


    push!(rows_table2, (
        dataset = dataset_label(dataset_name),
        TS = ts,
        IC = size_label(ic),
        KP = size_label(kp),
        model = get_model_string(model_name),
        mae_DL = create_output_string(dl_vs_tl[:mae_complete][:mean].rel_diff * 100),
        mae_ODE = create_output_string(ode_vs_tl[:mae_complete][:mean].rel_diff * 100),
        complement_pta_DL = create_output_string(dl_vs_tl[:complement_pta][:mean].rel_diff * 100),
        complement_pta_rel_ODE = create_output_string(ode_vs_tl[:complement_pta][:mean].rel_diff * 100),
        rmse_DL = create_output_string(dl_vs_tl[:rmse_complete][:mean].rel_diff * 100),
        rmse_rel_ODE = create_output_string(ode_vs_tl[:rmse_complete][:mean].rel_diff * 100)
    ))

    push!(rows_CI, (
        dataset = dataset_label(dataset_name),
        TS = ts,
        IC = size_label(ic),
        KP = size_label(kp),
        model = get_model_string(model_name),
        mae_DL = create_ci_string(ci_mae_dl_tl),
        mae_ODE = create_ci_string(ci_mae_ode_tl),
        complement_pta_DL = create_ci_string(ci_pta_dl_tl),
        complement_pta_rel_ODE = create_ci_string(ci_pta_ode_tl),
        rmse_DL = create_ci_string(ci_rmse_dl_tl),
        rmse_rel_ODE = create_ci_string(ci_rmse_ode_tl),
    ))

end

print("\n\n")

pretty_table(
    rows_table2,
    column_labels = ([
        ["Dataset", MultiColumn(3, "Best TL Run"), "Model", MultiColumn(2, "MAE"), MultiColumn(2, "1-PTA"), MultiColumn(2, "RMSE")],
        ["", "TS", "IC", "KP", "", "DL", "ODE", "DL", "ODE", "DL", "ODE"]
    ]),
    merge_column_label_cells = :auto,
    backend = :latex
)

pretty_table(
    rows_CI,
    column_labels = ([
        ["Dataset", MultiColumn(3, "Best TL Run"), "Model", MultiColumn(2, "MAE"), MultiColumn(2, "1-PTA"), MultiColumn(2, "RMSE")],
        ["", "TS", "IC", "KP", "", "DL", "ODE", "DL", "ODE", "DL", "ODE"]
    ]),
    merge_column_label_cells = :auto,
    backend = :latex
)

