### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ 9de3bb8a-e9d4-11ed-20c9-35cadb6e8416
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 8c7133f0-1b79-4317-8349-4609b9447793
using MLJBase

# ╔═╡ bf3b1a57-fd8b-4858-9770-7edc239242e4
using MLJScikitLearnInterface: RidgeCVClassifier

# ╔═╡ 0bbd9aab-8d32-4854-af47-074f679d050c
using StatisticalMeasures

# ╔═╡ 0c229a07-b0ba-4afb-a5c5-2679b43f2d91
using TimeSeriesClassification: MiniRocketModel, DataSets

# ╔═╡ 3dd04ec5-4b0b-451a-b44c-982d55d5f750
using TimeSeriesClassification.DataSets.Loaders: UCRArchive

# ╔═╡ d8ba75a0-717b-4c33-80d2-9287449ca29e
md"
# MiniRocket example
"

# ╔═╡ 54acb425-c1db-4adb-83a8-91e90c0c9172
md"
## Init environment and packages
"

# ╔═╡ 3d1ee662-382a-442c-a8d7-87e7d4d88aee
md"
Let's activate the TimeSeriesClassification package environment first
"

# ╔═╡ 02ca3013-5db3-4657-a8c6-1f553ea76c23
md"
... and import required packages like MLJ or MLJBase and ScikitLearn for Ridge Regression
"

# ╔═╡ 213e0990-3e8c-4a83-ab79-d2ab54569fa9
md"
... and import TimeSeriesClassification packages
"

# ╔═╡ c3005465-cf63-4b1e-a9f7-1a88965a0d60
md"
## Load dataset
"

# ╔═╡ 8f7407a8-d933-4bdf-b78c-6b44535e0c1e
md"
Which datasets are even available?
"

# ╔═╡ 31c762bf-a417-44ff-9aac-8b363838485a
ucr_archive_datasets = DataSets.list_available_datasets(UCRArchive)

# ╔═╡ 7d71c7fa-04c8-4f80-8fe1-fbfac1386632
md"
Let's try Chinatown dataset. Don't forget to covnert the input data into matrices witch samples stored in colums! Julia is column-major and so is this implementation of MiniRocket.
"

# ╔═╡ 223fe44e-58c0-4e0e-8de6-0828a4a2c5c7
begin
	trainX, trainY, testX, testY = DataSets.load_dataset(UCRArchive, :Chinatown)
	trainX = DataSets.dataset_flatten_to_matrix(trainX)
	trainY = categorical(trainY)
	testX = DataSets.dataset_flatten_to_matrix(testX)
	testY = categorical(testY)
end;

# ╔═╡ 847c07f2-96fb-418b-a124-7522530f2241
md"
## Build model
"

# ╔═╡ 03d562a2-caa5-4882-a4c3-b602953fecca
md"
Now it's time to init MiniRocket model!
"

# ╔═╡ f52d9dca-4a6f-494a-bfac-e418a403f531
m = MiniRocketModel();

# ╔═╡ 261bc796-f94c-4d29-8053-01786c0be25e
md"
MiniRocket isn't a classifier but just a transofmer. Let's chain MiniRocket with Ridge Regression as recommended by MiniRocket's auhors for datasets with less samples than the number of generated features.
"

# ╔═╡ f2fb819f-1a64-4e2d-8412-5bfd88db1cb6
pipe = Pipeline(
	# This is MiniRocket	
	m,
	# Scikit-Learn's RidgeCVClassifier requires samples to be in rows, lets transpose the transformed result from MiniRocket
	(X) -> table(transpose(X)),
	RidgeCVClassifier()
)

# ╔═╡ 2ae9683a-9ac3-4c33-b29a-fdc96cc7dd12
mach = machine(pipe, transpose(trainX), trainY)

# ╔═╡ d7f36102-574e-4c92-9886-fd8da8a8af5d
md"
## Train
"

# ╔═╡ b58d07db-9aca-47b8-af3b-f8320960b1a8
md"
Time to train the model!
"

# ╔═╡ 284ba0bf-7478-47f0-8c64-af6fa7e389e3
fit!(mach)

# ╔═╡ 7fbffd74-f5a2-43bb-895f-6d23f929cfd5
md"
## Evaluate
"

# ╔═╡ e9fc1277-ecbc-49da-bef8-7cfb5374576a
pred = predict(mach, transpose(testX))

# ╔═╡ 9c6e3de1-de6c-4394-a316-81fcd4d6c870
accuracy(testY, pred)

# ╔═╡ Cell order:
# ╟─d8ba75a0-717b-4c33-80d2-9287449ca29e
# ╟─54acb425-c1db-4adb-83a8-91e90c0c9172
# ╟─3d1ee662-382a-442c-a8d7-87e7d4d88aee
# ╠═9de3bb8a-e9d4-11ed-20c9-35cadb6e8416
# ╟─02ca3013-5db3-4657-a8c6-1f553ea76c23
# ╠═8c7133f0-1b79-4317-8349-4609b9447793
# ╠═bf3b1a57-fd8b-4858-9770-7edc239242e4
# ╠═0bbd9aab-8d32-4854-af47-074f679d050c
# ╟─213e0990-3e8c-4a83-ab79-d2ab54569fa9
# ╠═0c229a07-b0ba-4afb-a5c5-2679b43f2d91
# ╠═3dd04ec5-4b0b-451a-b44c-982d55d5f750
# ╟─c3005465-cf63-4b1e-a9f7-1a88965a0d60
# ╟─8f7407a8-d933-4bdf-b78c-6b44535e0c1e
# ╠═31c762bf-a417-44ff-9aac-8b363838485a
# ╟─7d71c7fa-04c8-4f80-8fe1-fbfac1386632
# ╠═223fe44e-58c0-4e0e-8de6-0828a4a2c5c7
# ╟─847c07f2-96fb-418b-a124-7522530f2241
# ╟─03d562a2-caa5-4882-a4c3-b602953fecca
# ╠═f52d9dca-4a6f-494a-bfac-e418a403f531
# ╟─261bc796-f94c-4d29-8053-01786c0be25e
# ╠═f2fb819f-1a64-4e2d-8412-5bfd88db1cb6
# ╠═2ae9683a-9ac3-4c33-b29a-fdc96cc7dd12
# ╟─d7f36102-574e-4c92-9886-fd8da8a8af5d
# ╟─b58d07db-9aca-47b8-af3b-f8320960b1a8
# ╠═284ba0bf-7478-47f0-8c64-af6fa7e389e3
# ╟─7fbffd74-f5a2-43bb-895f-6d23f929cfd5
# ╠═e9fc1277-ecbc-49da-bef8-7cfb5374576a
# ╠═9c6e3de1-de6c-4394-a316-81fcd4d6c870
