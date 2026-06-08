import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  
from sklearn.naive_bayes import CategoricalNB     
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. Veri setinin yüklenmesi ve ön işleme aşaması
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(url, names=columns)

# Sözel verileri sayısal değerlere dönüştürme 
encoder = OrdinalEncoder()
X = encoder.fit_transform(df.drop("class", axis=1))
y = df["class"].map({"unacc": 0, "acc": 1, "good": 2, "vgood": 3}).values

# Veriyi %80 Eğitim, %20 Test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Grafiklerde kullanacağımız sınıf isimleri
class_names = ['Kabul Edilemez (unacc)', 'Kabul Edilebilir (acc)', 'İyi (good)', 'Çok İyi (vgood)']

# 2. Modellerin tanımlanması
models = {
    "Naive Bayes (Sizin Seçiminiz)": CategoricalNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Neural Network (MLP)": MLPClassifier(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

results = {}
saved_confusion_matrices = {}  

# 3. Modellerin eğitilmesi ve metriklerin hesaplanması
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    saved_confusion_matrices[name] = cm  

    accuracies, sensitivities, specificities, f_measures = [], [], [], []

    for i in range(4):  
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f_measure = (2 * (precision * sensitivity) / (precision + sensitivity) 
                     if (precision + sensitivity) > 0 else 0)

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f_measures.append(f_measure)

    results[name] = {
        "Accuracy": np.mean(accuracies) * 100,
        "Sensitivity (Recall)": np.mean(sensitivities) * 100,
        "Specificity": np.mean(specificities) * 100,
        "F-Measure": np.mean(f_measures) * 100,
    }

df_results = pd.DataFrame(results).T
print("\n--- MODEL KARŞILAŞTIRMA TABLOSU ---")
print(df_results.round(2))

# 4. Görselleştirme1: VERİ DAĞILIM GRAFİĞİ 
plt.figure(figsize=(9, 5))
ax = sns.countplot(x="class", data=df, palette="viridis", 
                   order=["unacc", "acc", "good", "vgood"])

# Çubukların üzerine gerçek sayıları (frekansları) yazdırma döngüsü
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=11, fontweight='bold')

plt.title("Car Evaluation Veri Seti - Hedef Sınıf Dağılım Grafiği (Ham Veri)", fontsize=13, fontweight='bold')
plt.xlabel("Araç Uygunluk Sınıfları", fontsize=11)
plt.ylabel("Frekans (Araç Adedi)", fontsize=11)
plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show() 


# Görselleştirme2: Performans metrikleri bar grafiği
df_results.plot(kind="bar", figsize=(12, 6))
plt.title("Naive Bayes ve Diğer Modellerin Performans Metrikleri Karşılaştırması", fontsize=14)
plt.ylabel("Yüzde Başarı (%)")
plt.xlabel("Algoritmalar")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

#  Görselleştirme3: Confusion Matrix
nb_cm = saved_confusion_matrices["Naive Bayes (Sizin Seçiminiz)"]
plt.figure(figsize=(8, 6))
sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Kategorik Naive Bayes Modeli - Karmaşıklık Matrisi (Confusion Matrix)', fontsize=12)
plt.ylabel('Gerçek Sınıf (Actual Class)')
plt.xlabel('Tahmin Edilen Sınıf (Predicted Class)')
plt.tight_layout()
plt.show()

#  Görselleştirme4: Naive Bayes çok sınıflı ROC eğrisi analizi
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_binarized.shape[1]
y_score = models["Naive Bayes (Sizin Seçiminiz)"].predict_proba(X_test)

plt.figure(figsize=(9, 7))
colors = ['purple', 'green', 'orange', 'red']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', label='Rastgele Tahmin (Diagonal Line)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
plt.title('Kategorik Naive Bayes Modeli - Çok Sınıflı ROC Eğrisi Analizi', fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
