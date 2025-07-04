import pandas as pd #Library to read excel files
import matplotlib.pyplot as plt #Library for visualizations
# Loading the Excel file
df = pd.read_excel("C:/Users/effie/OneDrive - Strathmore University/Documents/Mall_Customers.xlsx")
print(df.head())  #Viewing the data
print(df.info())

#Cleaning the Dataset
# Dropping the CustomerID column because it not useful for clustering, it is just an identifier
df.drop('CustomerID', axis=1, inplace=True)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}) # Encode 'Gender': Male = 0, Female = 1 since K-Means only works with numbers
df.dropna(inplace=True) # Dropping any rows with missing values (if present)

#Selecting the features we want to cluster on
#These are the variables that define customer behaviour
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

#Scaling the features to put them in the same level
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

#Running K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42) #Finding 3 customer groups
df['Cluster'] = kmeans.fit_predict(scaled_features) #Adding a new column in the dataset to show which group each customer belongs to

#Viewing the result
print(df.head())

#Visualizing in a scatter plot
plt.scatter(
    df['Annual Income (k$)'],           # X-axis
    df['Spending Score (1-100)'],       # Y-axis
    c=df['Cluster'],                    # Color each point by cluster group
    cmap='viridis',                     # Color theme
    s=60                                # Size of each dot
)

plt.title("Customer Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.grid(True)
plt.show()

# See average values for each group
print(df.groupby('Cluster').mean())

# Saving the data with clusters to Excel so the app can use it
df.to_excel("C:/Users/effie/OneDrive - Strathmore University/Documents/Mall_Customers.xlsx", index=False)
print("File saved!")
print(df.head())