#Kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

#Grafik gösterimde ayarlamalar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)
pd.set_option('display.width', 500)

#Büyük ölçekli veriseti
def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data

df = load_application_train()
df.head()
df.isnull().values.any()

df.isnull().sum()
df.isnull().sum().sum()
df.notnull().sum().sort_values(ascending = False)

#Küçük ölçekli veriseti
def load():
    data = pd.read_csv('datasets/titanic.csv')
    return data

df = load()
print(df['Age'].value_counts())
df.head()

#Aykırı değerleri yakalamadan önce boxplot grafiğini inceleme
sns.boxplot(x = df['Age'])
plt.show()

#Aykırı değerleri yakalamak
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1

low = q1 - 1.5 * iqr
up = q3 + 1.5 * iqr

#Aykırı değerlerin sorgulanması ve comprehension yapısı
df[(df['Age'] < low ) | (df['Age'] > up)]
df[(df['Age'] < low ) | (df['Age'] > up)].index
df[(df['Age'] < low ) | (df['Age'] > up)].any(axis = None)

#[age for age in df['Age'] if age < low | age > up]

#Yukarıdaki işlemleri fonksiyonlaştırmak

def outlier_thresholds(dataframe, column_name, q1 = 0.25, q3 = 0.75):
    q1 = dataframe[column_name].quantile(q1)
    q3 = dataframe[column_name].quantile(q3)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit

#Outlier değerlerini hesaplayan fonksiyon
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
   """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine nümerik görünümlü kategorik değişkenler de dahildir.
    :param dataframe: Değişken isimleri alınmak istenen dataframe
    :param cat_th: nümerik fakat kategorik değişkenler için sınıf eşik değeri
    :param car_th: kategorik fakat kardinal değişkenler için sınıf eşik değeri
    :return: cat_cols : list
    Kategorik değişken listesi
    num_cols : list
    Nümerik değişken listesi
    cat_but_car : list
    Kategorik gibi görünen ama mantıken nümeriğe daha yakın olan değişken listesi
    """

   cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
   num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes != 'O']
   cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == 'O']
   cat_cols = cat_cols + num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
   num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
   num_cols = [col for col in num_cols if col not in num_but_cat]

   print(f'Observations: {dataframe.shape[0]}')
   print(f'Variables: {dataframe.shape[1]}')
   print(f'cat_cols: {len(cat_cols)}')
   print(f'num_cols: {len(num_cols)}')
   print(f'cat_but_car: {len(cat_but_car)}')
   print(f'num_but_cat: {len(num_but_cat)}')

   return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#PassengerId değişkeni işlevsiz olduğu için nümerik değişkenlerden çıkardık
num_cols = [col for col in num_cols if col not in 'PassengerId']

#Outlier check
for col in num_cols:
    print(col, check_outlier(df, col))

def outlier_values(dataframe, col_name, index = False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].shape[0] > 10:
        print(dataframe.head())
    else:
        print(dataframe)
    if index == True:
        return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].index

#Veri setindeki outlierları kaldırmak
def remove_outliers(dataframe,col_name):
    low, up = remove_outliers(dataframe, col_name)
    without_outliers = dataframe[~((dataframe[col_name] > up) | (dataframe < low))]
    return without_outliers

def replace_with_borders(dataframe, col_name):
    low, up = check_outlier(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
    dataframe.loc[(dataframe[col_name] < low), col_name] = low
    return dataframe

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, 'carat')
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style ='.-')
plt.show()

th = np.sort(df_scores)[3]
df[df_scores < th].drop(axis = 0 , labels = df[df_scores < th].index)

df = load()
df.head()


#Missing Values fonksiyonu
def missing_tables(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe.isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending= False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss, ratio'])
    print(missing_df, end = '\n')
    if na_name:
        return na_columns

#Kategorik Değişken Kırılımında Değer Atama
#df[column].fillna(df.groupby(column)[column_to_transform].transform('...')
df['Age'].fillna(df.groupby('Sex')['Age'].transform('mean')).isnull().sum()

#Tahmine Dayalı Atama ile Doldurma
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in 'PassengerId']

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns = dff.columns)
dff.head()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns = dff.columns)

df['age_imputed_knn'] = dff[['Age']]
df.loc[df['Age'].isnull(), ['Age','age_imputed_knn']]

msno(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

na_cols = missing_tables(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN' : temp_df.groupby(col)[target].mean(),
                            'COUNT' : temp_df.groupby(col)[target].count()}), end = '\n\n\n')

missing_vs_target(df, 'Survived',na_cols)

#Label Encoder
df = load()
df.head()
df['Sex'].head()

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

two_class_categorical = [col for col in df.columns if df[col].dtype not in [int,float]
                         and col.nunique == 2]
label_encoder(df, two_class_categorical)

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns= categorical_cols, drop_first = drop_first)
    return dataframe

df = load()
ss = StandardScaler()
df['Age_std'] = ss.fit_transform(df[['Age']])

rs = RobustScaler()
df['Robost_Age'] = rs.fit_transform(df[['Age']])

