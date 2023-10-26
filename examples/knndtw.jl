### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ ecbdafa9-5bf2-4b26-9b0b-f0dd2a2f2d7e
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 00b0fe73-dfb0-44f9-a627-32a5778f16b5
using MLJBase

# ╔═╡ b9246117-3580-44a9-b272-7761b6357484
using StatisticalMeasures

# ╔═╡ 08bc10bd-76c4-4dba-ae55-7d6831363632
using TimeSeriesClassification: KNNDTWModel, DTWSakoeChiba, DataSets

# ╔═╡ 0532c6d2-5f69-45b2-a880-b5253396c4c0
using TimeSeriesClassification.DataSets.Loaders: UCRArchive

# ╔═╡ 6670cee1-7321-42eb-ac66-321d6a77773f
md"
# KNN+DTW example
"

# ╔═╡ 8eb1a757-1e2d-4320-86e2-603a209bce1c
md"
## Init environment and packages
"

# ╔═╡ 58bcd936-4b9e-4eaa-810c-49334f91f3df
md"
Let's activate the TimeSeriesClassification package environment first
"

# ╔═╡ 8cf09d70-28e5-4cec-a752-4a8016ec6520
md"
... and import required packages like MLJ or MLJBase
"

# ╔═╡ 5b5a7832-bd13-403e-9425-d31bab258473
md"
... and import TimeSeriesClassification packages
"

# ╔═╡ 83805247-d401-4796-b05b-a64c8263810c
md"
## Load dataset
"

# ╔═╡ b5322e9e-cb7e-4e7d-84a4-55196b0a9a50
md"
Which datasets are even available?
"

# ╔═╡ cdb0f86f-6e98-462b-be76-ebdebe82da67
ucr_archive_datasets = DataSets.list_available_datasets(UCRArchive)

# ╔═╡ cc11fe3b-71bb-4bd7-a3e3-9706ece94764
metadata_train, metadata_test = DataSets.load_dataset_metadata(UCRArchive, :Chinatown)

# ╔═╡ 50d6bf71-2ea4-4f65-9c20-b0486e1d00d9
md"
Let's try Chinatown dataset. Don't forget to covnert the input data into matrices witch samples stored in colums! Julia is column-major and so is this implementation of MiniRocket.
"

# ╔═╡ 20850c4b-220b-4e20-be9d-abd55035827e
begin
	trainX, trainY, testX, testY = DataSets.load_dataset(UCRArchive, :Chinatown)
	trainX = DataSets.dataset_flatten_to_matrix(trainX)
	trainY = categorical(trainY)
	testX = DataSets.dataset_flatten_to_matrix(testX)
	testY = categorical(testY)
end;

# ╔═╡ f8059271-540b-44fc-bbb1-c2694d175a54
md"
## Build model
"

# ╔═╡ b698c522-98eb-47a4-848d-84a68cecca22
md"
Now it's time to init KNN+DTW model!
"

# ╔═╡ 177ea151-0c0d-490d-ac95-bb99ad2159f2
knndtw = KNNDTWModel(
	K = 3,
	weights = :distance,
	distance = DTWSakoeChiba{eltype(trainX)}(radius=5)
)

# ╔═╡ 56ad9782-3349-4ad6-8a30-12aa8f322d0c
mach = machine(knndtw, transpose(trainX), trainY)

# ╔═╡ 64a0ffc2-22e8-43f7-a216-80337a9d6434
md"
Time to train the model!
"

# ╔═╡ cc059514-c242-48a1-81b1-d6f7593b2579
fit!(mach)

# ╔═╡ 37b6b67f-ac13-48ea-936e-487aabe6b631
md"
## Evaluate
"

# ╔═╡ 6341b366-b832-4a1e-8ffb-7a8b1e4691f6
yhat = predict_mode(mach, transpose(testX))

# ╔═╡ 9585942e-98f5-472f-92a1-3c9f23e96076
accuracy(testY, yhat)

# ╔═╡ Cell order:
# ╟─6670cee1-7321-42eb-ac66-321d6a77773f
# ╟─8eb1a757-1e2d-4320-86e2-603a209bce1c
# ╟─58bcd936-4b9e-4eaa-810c-49334f91f3df
# ╠═ecbdafa9-5bf2-4b26-9b0b-f0dd2a2f2d7e
# ╟─8cf09d70-28e5-4cec-a752-4a8016ec6520
# ╠═00b0fe73-dfb0-44f9-a627-32a5778f16b5
# ╠═b9246117-3580-44a9-b272-7761b6357484
# ╟─5b5a7832-bd13-403e-9425-d31bab258473
# ╠═08bc10bd-76c4-4dba-ae55-7d6831363632
# ╠═0532c6d2-5f69-45b2-a880-b5253396c4c0
# ╟─83805247-d401-4796-b05b-a64c8263810c
# ╟─b5322e9e-cb7e-4e7d-84a4-55196b0a9a50
# ╠═cdb0f86f-6e98-462b-be76-ebdebe82da67
# ╠═cc11fe3b-71bb-4bd7-a3e3-9706ece94764
# ╟─50d6bf71-2ea4-4f65-9c20-b0486e1d00d9
# ╠═20850c4b-220b-4e20-be9d-abd55035827e
# ╟─f8059271-540b-44fc-bbb1-c2694d175a54
# ╟─b698c522-98eb-47a4-848d-84a68cecca22
# ╠═177ea151-0c0d-490d-ac95-bb99ad2159f2
# ╠═56ad9782-3349-4ad6-8a30-12aa8f322d0c
# ╟─64a0ffc2-22e8-43f7-a216-80337a9d6434
# ╠═cc059514-c242-48a1-81b1-d6f7593b2579
# ╟─37b6b67f-ac13-48ea-936e-487aabe6b631
# ╠═6341b366-b832-4a1e-8ffb-7a8b1e4691f6
# ╠═9585942e-98f5-472f-92a1-3c9f23e96076
