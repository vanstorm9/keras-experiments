import pandas
import matplotlib.pyplot as plt

# We are not interested in the date within the dataset, therefore we can exclude it in the first column
# Furthermore, the dataset contains footer information we can ignore

dataset = pandas.read_csv('data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

print 'Dataset:', dataset.shape

plt.plot(dataset)
plt.show()
