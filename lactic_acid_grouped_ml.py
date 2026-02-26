import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # สำหรับบันทึกโมเดลไว้ใช้งานภายหลัง
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

# 1. จัดการคำเตือนและระบบพื้นฐาน
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# นำเข้าอัลกอริทึมให้ครบถ้วนทุกตระกูล
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              HistGradientBoostingRegressor, StackingRegressor, VotingRegressor)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def perform_eda(df, target_col):
    """ฟังก์ชันสำหรับการทำ Exploratory Data Analysis (EDA) และบันทึกรูปภาพ"""
    print("\n" + "="*60)
    print(" 🔍 เริ่มกระบวนการ Exploratory Data Analysis (EDA) ")
    print("="*60)
    
    # 1. การกระจายตัวของตัวแปรเป้าหมายและ Correlation Heatmap
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True, color='blue')
    plt.title(f'Distribution of {target_col}')
    
    plt.subplot(1, 2, 2)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('eda_distribution_and_correlation.png', dpi=300)
    print("💾 บันทึกรูปภาพ: eda_distribution_and_correlation.png")
    plt.show()

    # 2. ตรวจสอบความสัมพันธ์ระหว่าง Carbon Concentration กับ Lactic Acid รายสายพันธุ์
    if 'C_Conc (g/L)' in df.columns:
        plt.figure(figsize=(14, 8))
        n_strains = df['Strain'].nunique()
        # ใช้ husl palette เพื่อให้สีตัดกันชัดเจนแม้มีจำนวน Strain มาก
        custom_palette = sns.color_palette("husl", n_colors=n_strains)
        
        sns.scatterplot(
            data=df, 
            x='C_Conc (g/L)', 
            y=target_col, 
            hue='Strain', 
            palette=custom_palette, 
            alpha=0.8, 
            s=120, 
            edgecolor='black',
            style='Strain'  # เพิ่มรูปทรง Marker เพื่อช่วยแยกแยะ
        )
        plt.title(f'C_Conc vs Lactic Acid Yield ({n_strains} Strains Visualized)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', title='Strains')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('eda_carbon_vs_lactic.png', dpi=300)
        print("💾 บันทึกรูปภาพ: eda_carbon_vs_lactic.png")
        plt.show()

def plot_imputation_comparison(df_before, df_after, num_cols):
    """กราฟเปรียบเทียบจำนวนข้อมูลและการกระจายตัวก่อนและหลังเติมค่าว่าง"""
    plt.figure(figsize=(14, 7))
    counts_before = df_before[num_cols].count()
    counts_after = df_after[num_cols].count()
    
    comp_df = pd.DataFrame({
        'Feature': num_cols,
        'Before Imputation': counts_before.values,
        'After Imputation': counts_after.values
    }).melt(id_vars='Feature', var_name='Status', value_name='Count')
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=comp_df, x='Count', y='Feature', hue='Status', palette='Set2')
    plt.title('Data Completeness (Before vs After Imputation)')
    
    # ดูการกระจายตัวของ Carbon Concentration เป็นตัวอย่างหลัก
    if 'C_Conc (g/L)' in num_cols:
        plt.subplot(1, 2, 2)
        sns.kdeplot(df_before['C_Conc (g/L)'].dropna(), label='Before (Original)', fill=True, color='red', alpha=0.3)
        sns.kdeplot(df_after['C_Conc (g/L)'], label='After (Imputed)', fill=True, color='blue', alpha=0.3)
        plt.title('Distribution Comparison: C_Conc (g/L)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_imputation_comparison.png', dpi=300)
    print("💾 บันทึกรูปภาพ: data_imputation_comparison.png")
    plt.show()

def smoothed_target_encoding(train_df, test_df, group_col, target_col, m=10):
    """คำนวณ Target Encoding แบบ Smoothing ป้องกัน Data Leakage"""
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(group_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * global_mean) / (counts + m)
    train_enc = train_df[group_col].map(smooth).fillna(global_mean)
    test_enc = test_df[group_col].map(smooth).fillna(global_mean)
    return train_enc, test_enc, smooth, global_mean

def predict_yield(model, smooth_map, global_mean, input_data):
    """ฟังก์ชันทำนายผลผลิตสำหรับข้อมูลใหม่"""
    new_df = pd.DataFrame([input_data])
    new_df['CN_Ratio'] = new_df['C_Conc (g/L)'] / new_df['N_Conc (g/L)'].replace(0, np.nan)
    new_df['CN_Ratio'] = new_df['CN_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    new_df['C_per_Time'] = new_df['C_Conc (g/L)'] / new_df['Time (h)'].replace(0, np.nan)
    new_df['C_per_Time'] = new_df['C_per_Time'].fillna(0)
    new_df['Temp_pH'] = new_df['Temp (°C)'] * new_df['pH']
    new_df['Strain_Enc'] = new_df['Strain'].map(smooth_map).fillna(global_mean)
    return model.predict(new_df)[0]

def optimize_lab_conditions(model, df, smooth_map, global_mean, target_yield=90):
    """จำลองหาเงื่อนไขที่คุ้มค่าที่สุด (High Yield, Low Time/C/N) เพื่อแนะนำห้องแล็บ"""
    print("\n" + "="*60)
    print(f" 🧪 เริ่มกระบวนการหาเงื่อนไขที่เหมาะสม (Target Yield > {target_yield} g/L) ")
    print("="*60)
    
    strains = df['Strain'].unique()
    c_sources = df['Carbon Source'].unique()
    n_sources = df['N_Source'].unique()
    modes = df['Mode (รูปแบบ)'].unique()
    aerations = df['Aeration'].unique()
    
    results = []
    for _ in range(3000): # เพิ่มจำนวนรอบ Simulation
        c_conc = np.random.uniform(df['C_Conc (g/L)'].min(), df['C_Conc (g/L)'].max())
        n_conc = np.random.uniform(df['N_Conc (g/L)'].min(), df['N_Conc (g/L)'].max())
        time = np.random.uniform(df['Time (h)'].min(), df['Time (h)'].max())
        temp = np.random.choice([30, 37, 40, 45])
        ph = np.random.uniform(5.0, 7.0)
        
        trial = {
            'Strain': np.random.choice(strains),
            'Carbon Source': np.random.choice(c_sources),
            'C_Conc (g/L)': round(c_conc, 2),
            'N_Source': np.random.choice(n_sources),
            'N_Conc (g/L)': round(n_conc, 2),
            'Mode (รูปแบบ)': np.random.choice(modes),
            'Agitation (rpm)': np.random.choice([0, 100, 200]),
            'Aeration': np.random.choice(aerations),
            'DO / Gas Flow': 0,
            'Temp (°C)': temp,
            'pH': round(ph, 1),
            'Time (h)': round(time, 1)
        }
        
        pred = predict_yield(model, smooth_map, global_mean, trial)
        if pred >= target_yield:
            # Efficiency Score: เน้น Yield สูง แต่ใช้สารอาหารและเวลาต่ำที่สุด
            efficiency = pred / (trial['Time (h)'] * trial['C_Conc (g/L)'] * (trial['N_Conc (g/L)'] + 1)) * 1000
            trial['Predicted Yield'] = round(pred, 2)
            trial['Efficiency Score'] = round(efficiency, 4)
            results.append(trial)
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        top_recommendations = res_df.sort_values(by='Efficiency Score', ascending=False).head(5)
        print(f"พบ {len(res_df)} สภาวะที่ให้ผลผลิตตามเป้าหมาย")
        print("\n[ข้อเสนอแนะ 5 อันดับแรกเพื่อประหยัดทรัพยากร (Top 5 Efficient Lab Settings)]")
        print(top_recommendations[['Strain', 'Carbon Source', 'C_Conc (g/L)', 'N_Conc (g/L)', 'Time (h)', 'Predicted Yield', 'Efficiency Score']])
        top_recommendations.to_csv('lab_optimization_recommendations.csv', index=False)
        print("\n💾 บันทึกข้อเสนอแนะสำเร็จในไฟล์: lab_optimization_recommendations.csv")
    else:
        print("❌ ไม่พบสภาวะที่ตรงตามเป้าหมายจากการจำลอง")

def process_and_train_final_model(filepath):
    # 2. โหลดและจัดการค่าว่าง
    df = pd.read_csv(filepath, na_values='-')
    df = df.drop(columns=['ID', 'LAB?'], errors='ignore')
    target_col = 'Lactic Acid (g/L)'
    df = df.dropna(subset=[target_col])

    # EDA ขั้นต้น
    perform_eda(df, target_col)
    df_before = df.copy()

    # ระบุคอลัมน์ตัวเลขสำหรับการ Impute
    num_cols_raw = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols_raw: num_cols_raw.remove(target_col)
    
    # ดำเนินการ Grouped Imputation (เติมค่าตามสายพันธุ์)
    for col in num_cols_raw:
        df[col] = df[col].fillna(df.groupby('Strain')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median())

    # กราฟเปรียบเทียบสถิติก่อน-หลังเติมค่าว่าง
    plot_imputation_comparison(df_before, df, num_cols_raw)

    # สถิติตัวเลขสรุป
    print("\n[สรุปสถิติหลังจัดการข้อมูล (After Preprocessing Summary)]")
    print(df.describe().transpose()[['count', 'mean', 'std', 'min', 'max']])

    # 3. เตรียมข้อมูลสำหรับการเทรน
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Engineering
    def engineer_features(data):
        df_eng = data.copy()
        df_eng['CN_Ratio'] = df_eng['C_Conc (g/L)'] / df_eng['N_Conc (g/L)'].replace(0, np.nan)
        df_eng['CN_Ratio'] = df_eng['CN_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(df_eng['CN_Ratio'].median())
        df_eng['C_per_Time'] = df_eng['C_Conc (g/L)'] / df_eng['Time (h)'].replace(0, np.nan)
        df_eng['C_per_Time'] = df_eng['C_per_Time'].fillna(df_eng['C_per_Time'].median())
        df_eng['Temp_pH'] = df_eng['Temp (°C)'] * df_eng['pH']
        return df_eng

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Target Encoding สำหรับสายพันธุ์
    train_temp = X_train.copy()
    train_temp[target_col] = y_train
    X_train['Strain_Enc'], X_test['Strain_Enc'], smooth_map, g_mean = smoothed_target_encoding(
        train_temp, X_test, 'Strain', target_col, m=10
    )

    # 5. Preprocessing Pipeline
    # กำหนดตัวแปรฟีเจอร์ตัวเลขที่จะใช้ใน Pipeline
    new_num_cols = num_cols_raw + ['CN_Ratio', 'Strain_Enc', 'C_per_Time', 'Temp_pH']
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())
            ]), new_num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ])

    # 6. รายการโมเดล (รวบรวมครบถ้วน 12 รูปแบบ)
    base_rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    top_base_models = [
        ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance'))
    ]

    models_to_test = {
        'Ridge': (Ridge(max_iter=10000), {'regressor__alpha': [1.0, 10.0]}),
        'ElasticNet': (ElasticNet(max_iter=10000, random_state=42), {'regressor__alpha': [0.1, 1.0]}),
        'SVR': (SVR(), {'regressor__C': [1.0, 10.0, 100.0]}),
        'KNN': (KNeighborsRegressor(), {'regressor__n_neighbors': [3, 5], 'regressor__weights': ['distance']}),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {'regressor__max_depth': [5, 10]}),
        'RandomForest': (RandomForestRegressor(random_state=42), {'regressor__n_estimators': [200], 'regressor__max_depth': [10]}),
        'GradientBoosting': (GradientBoostingRegressor(random_state=42), {'regressor__n_estimators': [300, 500], 'regressor__learning_rate': [0.05, 0.1]}),
        'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=42), {'regressor__learning_rate': [0.05]}),
        'Stacking_Ensemble': (StackingRegressor(estimators=top_base_models, final_estimator=Ridge(alpha=1.0), cv=5), {}),
        'Weighted_Voting': (VotingRegressor(estimators=top_base_models, weights=[2, 1, 1]), {})
    }

    if HAS_XGB:
        models_to_test['XGBoost'] = (XGBRegressor(random_state=42), {'regressor__n_estimators': [200]})
    if HAS_LGBM:
        models_to_test['LightGBM'] = (LGBMRegressor(random_state=42, verbose=-1), {'regressor__n_estimators': [100]})

    # 7. Benchmarking
    results = []
    best_overall_r2 = -np.inf
    best_overall_model = None
    winner_name = ""
    cv_strategy = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    print("\n" + "-"*75)
    print(f"{'Algorithm':<20} | {'CV R2':<10} | {'Test R2':<10} | {'MAE':<10}")
    print("-" * 75)

    for name, (model_obj, params) in models_to_test.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selection', SelectFromModel(base_rf_selector, threshold="median")),
            ('regressor', model_obj)
        ])
        grid = GridSearchCV(pipe, params, cv=cv_strategy, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({'Algorithm': name, 'CV_R2': grid.best_score_, 'Test_R2': test_r2, 'MAE': mae})
        print(f"{name:<20} | {grid.best_score_:.4f}     | {test_r2:.4f}     | {mae:.2f}")

        if test_r2 > best_overall_r2:
            best_overall_r2 = test_r2
            best_overall_model = grid.best_estimator_
            winner_name = name

    # 8. วิเคราะห์ Performance และบันทึกผล
    y_final_pred = best_overall_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_final_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
    final_mape = mean_absolute_percentage_error(y_test, y_final_pred)

    print("\n" + "="*50)
    print(f" 🏆 ผู้ชนะ: {winner_name} ")
    print("="*50)
    print(f"R-squared:    {best_overall_r2:.4f}")
    print(f"MAE:          {final_mae:.2f} g/L")
    print(f"RMSE:         {final_rmse:.2f} g/L")
    print(f"MAPE:         {final_mape*100:.2f} %")
    print("="*50)

    # 9. ส่งออกไฟล์ .pkl เพื่อการใช้งานภายหลัง
    model_export = {
        'model': best_overall_model,
        'smooth_map': smooth_map,
        'global_mean': g_mean,
        'winner_name': winner_name,
        'performance': {
            'r2': best_overall_r2,
            'mae': final_mae,
            'mape': final_mape
        }
    }
    joblib.dump(model_export, 'lactic_acid_best_model.pkl')
    print("\n💾 ส่งออกไฟล์โมเดลสำเร็จ: lactic_acid_best_model.pkl")

    # 10. พลอตกราฟเปรียบเทียบ (พร้อมตัวเลข Accuracy)
    res_df = pd.DataFrame(results).sort_values(by='Test_R2', ascending=False)
    plt.figure(figsize=(15, 8))
    res_melted = res_df.melt(id_vars='Algorithm', value_vars=['CV_R2', 'Test_R2'])
    ax = sns.barplot(data=res_melted, x='Algorithm', y='value', hue='variable', palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, rotation=45, size=9)
    plt.title('Final Model Benchmarking (Accuracy Scores)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('model_benchmark_results.png', dpi=300)
    plt.show()

    # ทำการ Optimize หาเงื่อนไขที่คุ้มค่าที่สุด
    optimize_lab_conditions(best_overall_model, df, smooth_map, g_mean, target_yield=90)

    return best_overall_model, smooth_map, g_mean, winner_name

if __name__ == "__main__":
    try:
        final_model, s_map, g_mean, winner = process_and_train_final_model('20260224_Lactic_Acid.csv')
    except Exception as e:
        print(f"Error: {e}")