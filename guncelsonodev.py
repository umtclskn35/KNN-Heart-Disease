import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns

# Veri setini indirme
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
heart_disease = pd.read_csv(url, names=names, na_values='?')

X = heart_disease.drop('num', axis=1)  # features
y = heart_disease['num']               # targets

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# KNN işlemleri 6 adet komşu seçili
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Korelasyon matrisi
corrmat = heart_disease.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
plt.title("Korelasyon matrisi")
sns.heatmap(heart_disease[top_corr_features].corr(),annot=True,cmap='RdYlGn')
plt.show()

test_score = knn.score(X_scaled, y)
print(f"Test Seti Doğruluğu: {test_score}")

# En iyi değeri bulmak için bütün 1den  20 ye bütün komşu değerleri dener
score_list=[]
for each in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(X_scaled,y)
    score_list.append(knn.score(X_scaled,y))
plt.title("En Uygun Komşu Sayısı")
plt.plot(range(1,20),score_list)
plt.xlabel("Komşu Sayısı")
plt.ylabel("Doğruluk Oranı")
plt.show()

root = tk.Tk()
root.title("Kalp Hastalığı Tahmini")

labels = ['Yaş', 'Cinsiyet (0: Kadın, 1: Erkek)', 'Göğüs Ağrısı Türü', 'İstirahat Kan Basıncı', 'Serum Kolesterolü', 
          'Açlık Kan Şekeri (> 120 mg/dl)', 'İstirahat EKG Sonuçları', 'Maksimum Kalp Atış Hızı', 
          'Egzersizle İlişkili Anjina', 'Egzersizle İndüklenen ST Depresyonu', 'Egzersiz Zirve ST Segmentinin Eğimi', 
          'Floroskopi ile Boyanan Ana Damar Sayısı', 'Talasemi']
entries = []
hints = ['Yaş (age): 29 ile 77 arasında değişen bir sayı. Hastanın yaşını belirtir.', '(0: Kadın, 1: Erkek)', '(1: Tip1, 2: Tip2, 3: Tip3)', 'Dinlenme kan basıncı (trestbps): 94 ile 200 arasında değişen bir sayı. Kan basıncını mmHg cinsinden belirtir.', 'Kolestrol seviyesi (chol): 126 ile 564 arasında değişen bir sayı. Kolestrol miktarını mg/dL cinsinden belirtir.', 
         '(0: Hayır, 1: Evet)', 'Dinlenme elektrokardiyografik sonuçları (restecg): 0, 1 veya 2. Kalp elektrik aktivitesinin durumunu belirtir.', '(örnek: 160)', '(0: Hayır, 1: Evet)', 'Egzersizle indüklenen ST-depresyonu (oldpeak): 0 ile 6.2 arasında değişen bir sayı. Egzersizle indüklenen ST-depresyonunun derinliğini belirtir.', 
         'Eğim (slope): 1, 2 veya 3. ST-segmentinin eğimini belirtir.', 'CA (ca): 0, 1, 2 veya 3. Renksiz sayılarla belirtilen major damarların sayısını belirtir.', 'Thal (thal): 3, 6 veya 7. Kalp hastalığının tipini belirtir.']
for i, (label, hint) in enumerate(zip(labels, hints)):
    tk.Label(root, text=label).grid(row=i, column=0, padx=5, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    tk.Label(root, text=hint).grid(row=i, column=2, padx=5, pady=5)
    entries.append(entry)

def predict_heart_disease():
    # Giriş kutularından değerleri al
    values = [float(entry.get()) for entry in entries]

    # Alınan verileri bir numpy dizisine dönüştür
    data = np.array([values])

    # Veriyi standartlaştır
    data_scaled = scaler.transform(data)

    # Tahmin yap
    prediction = knn.predict(data_scaled)

    # Sonucu kullanıcıya göster
    if prediction[0] == 0:
        messagebox.showinfo("Sonuç", "Bu kişi kalp hastası değil.")
    else:
        messagebox.showinfo("Sonuç", "Bu kişi kalp hastası.")

predict_button = tk.Button(root, text="Tahmin Et", command=predict_heart_disease)
predict_button.grid(row=len(labels), columnspan=3, padx=5, pady=10)

# Arayüzü başlatma
root.mainloop()