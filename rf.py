import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('adult_preprocessed.csv')

#Giriş (X) ve çıkışın (y) belirlenmesi
X=df.drop('income',axis=1)
y=df['income']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#max_depth=12 ve min_samples_split=5 ile modelin ezber yapması zorlatırılır
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    min_samples_split=5, 
    random_state=42
)

#Random Forest Modelini Oluşturma ve Eğitme
#Modeli eğitme
model.fit(X_train, y_train)

y_train_pred=model.predict(X_train)     #Train verisi ile performans tahmini
y_test_pred=model.predict(X_test)    #Test verisi ile performans tahmini


print(f"Eğitim Verisi Başarı Oranı (Train Accuracy): {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Verisi Başarı Oranı (Test Accuracy): {accuracy_score(y_test, y_test_pred):.4f}")


#print("Model Başarı Oranı (Accuracy):", accuracy_score(y_test, y_pred))
print("\nDetaylı Rapor (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_test_pred))
