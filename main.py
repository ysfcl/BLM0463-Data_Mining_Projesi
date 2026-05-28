#DATA PREPROCESSING 

import pandas as pd
import numpy as np

# Kütüphane importları gerçekleştirilir

# veride virgüllerden sonra boşluklar bulunduğu için 'skipinitialspace' kullanılır
 
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

df=pd.read_csv('adult.data',names=columns,skipinitialspace=True)


print(f"Temizlik öncesi satır sayısı: {len(df)}")   #Temizlik öncesi: 32561


# Veri içerisindeki ? olan yerler NaN (boş eleman) olacak şekilde değiştirilir 
df=df.replace("?",np.nan)


# Temizleme işlemi yapılır ve NaN bulunan satırlar silinir
df_cleaned=df.dropna()


print(f"Temizlik sonrası satır sayısı: {len(df_cleaned)}")  #Temizlik sonrası: 30162


# Modelin anlayabilmesi için <=50K değerler 0, >50K değerleri 1 yapılır
df_cleaned['income'] = df_cleaned['income'].apply(lambda x: 1 if x == '>50K' else 0).astype(int)

# One Hot Encoding gerçekleştirilir
# workclass, education gibi metin tabanlı sütunları 0 ve 1'lerden oluşan sütunlara çevirir.
df_final = pd.get_dummies(df_cleaned)


# Yeni oluşturulan CSV dosyası kaydedilir
df_final.to_csv('adult_preprocessed.csv',index=False)