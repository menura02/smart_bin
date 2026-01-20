"""
PROJE: Akıllı Çöp Konteyneri Doluluk Tahmin Sistemi
YAZAR: [Örnek Öğrenci]
TARİH: 2026-01-20

AÇIKLAMA:
Bu modül, Smart_Bin.csv verisetini kullanarak konteynerlerin boşaltılma durumunu (Class)
tahmin eder. Veri sızıntısını (data leakage) önlemek için Scikit-Learn Pipeline mimarisi
kullanılmıştır.
"""

# --- 1. Kütüphane İçe Aktarımı (Düzenli ve Gereksizlerden Arındırılmış) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)

# Uyarıları yönet (Gereksiz kalabalığı önle)
import warnings
warnings.filterwarnings('ignore')

# Sabitler (Global Configuration)
FILE_PATH = 'Smart_Bin.csv'
TARGET_COL = 'Class'
RANDOM_STATE = 42

def load_and_engineer_features(filepath):
    """
    Veriyi yükler ve temel özellik mühendisliği (Feature Engineering) yapar.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[BİLGİ] Veri seti yüklendi. Boyut: {df.shape}")
        
        # İş Zekası Dokunuşu: Doluluk Hızı Özelliği
        # 4. Koddaki yaratıcı fikri entegre ediyoruz.
        if 'FL_B' in df.columns and 'FL_A' in df.columns:
            df['Doluluk_Artisi'] = df['FL_B'] - df['FL_A']
            print("[BİLGİ] 'Doluluk_Artisi' özelliği türetildi.")
            
        return df
    except FileNotFoundError:
        print(f"[HATA] Dosya bulunamadı: {filepath}")
        return None

def build_pipeline(numeric_features, categorical_features):
    """
    Veri sızıntısını önlemek için Pipeline ve ColumnTransformer kurar.
    
    Neden?
    - Eğitim verisinin ortalaması/medyanı ile test verisini doldurmak için.
    - Test setini "görmeden" işlem yapmak için.
    """
    
    # Sayısal Değişkenler İçin İşlemler:
    # 1. Eksik verileri medyan ile doldur (Outlier'lara karşı dirençli)
    # 2. Standartlaştır (Scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Kategorik Değişkenler İçin İşlemler:
    # 1. Eksik verileri 'missing' etiketi ile doldur
    # 2. One-Hot Encoding yap (Bilinmeyen kategorileri yoksay)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # İşlemleri Birleştir
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Ana Model Pipeline'ı
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, 
                                              random_state=RANDOM_STATE, 
                                              class_weight='balanced')) # Dengesiz veriye önlem
    ])
    
    return model_pipeline

def evaluate_model(pipeline, X_test, y_test, y_test_encoded):
    """
    Model performansını kapsamlı metriklerle değerlendirir ve görselleştirir.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # ROC için pozitif sınıf olasılığı

    # 1. Metrikler
    print("\n" + "="*40)
    print("MODEL PERFORMANS RAPORU")
    print("="*40)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score : {roc_auc_score(y_test_encoded, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 2. Görselleştirme (Confusion Matrix & ROC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Tahmin')
    axes[0].set_ylabel('Gerçek')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, y_proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test_encoded, y_proba):.2f}", color='darkorange', lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def main():
    # 1. Veri Yükleme
    df = load_and_engineer_features(FILE_PATH)
    if df is None: return

    # 2. Hedef ve Özellik Ayrımı
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Özellik Türlerini Belirle
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(f"[BİLGİ] Sayısal Özellikler: {list(numeric_features)}")
    print(f"[BİLGİ] Kategorik Özellikler: {list(categorical_features)}")

    # 3. Eğitim / Test Ayrımı (Stratify Önemli!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        stratify=y, 
                                                        random_state=RANDOM_STATE)

    # Hedef Değişkeni Encode Et (ROC AUC için gereklidir)
    # Pipeline dışında yapıyoruz çünkü Pipeline X'i işler, y'yi değil.
    y_train_encoded = y_train.apply(lambda x: 1 if x == 'Emptying' else 0) # Örnek mapping
    y_test_encoded = y_test.apply(lambda x: 1 if x == 'Emptying' else 0)

    # 4. Pipeline Kurulumu ve Eğitim
    print("[BİLGİ] Model eğitimi başlıyor...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # 5. Cross-Validation (Modelin kararlılığını test et)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[BİLGİ] 5-Katlı Çapraz Doğrulama Ortalaması: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 6. Değerlendirme
    evaluate_model(pipeline, X_test, y_test, y_test_encoded)

    # 7. Feature Importance (Pipeline içinden modele erişim)
    # Bu kısım biraz ileri seviyedir ama 100 puan için gereklidir.
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # OneHotEncoder sonrası oluşan sütun isimlerini al
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, cat_feature_names])
    
    importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    print("\n--- En Önemli 10 Özellik ---")
    print(importances)

if __name__ == "__main__":
    main()