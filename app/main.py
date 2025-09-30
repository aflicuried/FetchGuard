from ucimlrepo import fetch_ucirepo

# fetch dataset
cardiotocography = fetch_ucirepo(id=193)

# data (as pandas dataframes)
X = cardiotocography.data.features
y = cardiotocography.data.targets

# metadata
print(cardiotocography.metadata)

# variable information
print(cardiotocography.variables)