import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df=pd.read_csv('adult_preprocessed.csv')

#Giriş (X) ve çıkışın (y) belirlenmesi
X=df.drop('income',axis=1)  #income sütunu hedef değişken olduğu için X'ten çıkarılır
y=df['income']      #sadece income sütunu hedef değişken olarak alınır

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
print("\n\n Grafik oluşturuluyor...")

# ---CONFUSION (Karışıklık) MATRIX---
# Modelin kaç tahmini doğru, kaçını yanlış yaptığını matris olarak hesaplanır
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
# Seaborn (sns) ile mavi ısı haritası çizdiriliyor
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K (0)', '>50K (1)'], 
            yticklabels=['<=50K (0)', '>50K (1)'])

plt.title('Confusion Matrix')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.tight_layout()

# Grafiği proje klasörüne resim olarak kaydedilir
plt.savefig('confusion_matrix.png', dpi=300) 
plt.show() # Grafik ekranda gösterilir


# ---FEATURE IMPORTANCE (DEĞİŞKEN ÖNEM SIRASI)---
# Modelin geliri tahmin ederken en çok hangi 10 sütuna baktığını hesaplar
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
# Renkli bir çubuk grafik çizilir
sns.barplot(x=importances.values, y=importances.index, palette='viridis')

plt.title('Random Forest - En Önemli 10 Özellik (Feature Importance)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler (Features)')
plt.tight_layout()

# Grafik proje klasörüne resim olarak kaydedilir
plt.savefig('feature_importance.png', dpi=300) 
plt.show() # Grafik ekranda gösterilir

print("İşlem tamamlandı! 'confusion_matrix.png' ve 'feature_importance.png' klasörünüze kaydedildi.")