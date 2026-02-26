from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = 'lactic_acid_best_model.pkl'

try:
    # โหลดไฟล์ pkl
    loaded_obj = joblib.load(MODEL_PATH)
    # ตรวจสอบว่าเป็น dict หรือตัวโมเดลโดยตรง
    model = loaded_obj['model'] if isinstance(loaded_obj, dict) else loaded_obj
    print(f"✅ SMART BRAIN READY: {MODEL_PATH}")
    
    # ดึงรายชื่อฟีเจอร์ที่โมเดลต้องการ (16 ฟีเจอร์)
    if hasattr(model, 'feature_names_in_'):
        EXPECTED_FEATURES = list(model.feature_names_in_)
        print(f"📋 Model expects {len(EXPECTED_FEATURES)} features.")
    else:
        # Fallback กรณีหาคุณสมบัติไม่เจอ
        EXPECTED_FEATURES = [
            'Strain', 'Carbon Source', 'C_Conc (g/L)', 'N_Source', 'N_Conc (g/L)',
            'Mode (รูปแบบ)', 'Agitation (rpm)', 'Aeration', 'DO / Gas Flow',
            'Temp (°C)', 'pH', 'Time (h)', 'CN_Ratio', 'C_per_Time', 'Temp_pH', 'Strain_Enc'
        ]
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'model': MODEL_PATH})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- 1. รับค่าพื้นฐานจากหน้าเว็บ ---
        c_val = float(data.get('cConc', 60))
        n_val = float(data.get('nConc', 25))
        t_val = float(data.get('temp', 37))
        p_val = float(data.get('pH', 6.0))
        time_val = float(data.get('time', 36))
        
        # --- 2. คำนวณ Derived Features (เพื่อให้ครบ 16 ตัวตามโมเดล) ---
        cn_ratio = c_val / n_val if n_val > 0 else 0
        c_per_time = c_val / time_val if time_val > 0 else 0
        temp_ph = t_val * p_val
        
        # Strain Encoding แบบง่าย (ควรตรงกับที่ใช้เทรน)
        strain_name = data.get('strain', 'Lc. lactis IO-1')
        strain_list = ['B. coagulans', 'E. hirae', 'L. casei', 'L. delbrueckii', 'Lc. lactis IO-1']
        strain_enc = strain_list.index(strain_name) if strain_name in strain_list else 0

        # --- 3. รวบรวมข้อมูลให้ตรงตามชื่อคอลัมน์ที่โมเดลต้องการ ---
        raw_input = {
            'Strain': strain_name,
            'Carbon Source': data.get('carbonSource', 'Glucose (Syn)'),
            'C_Conc (g/L)': c_val,
            'N_Source': data.get('nitrogenSource', 'CSL+YE+Pep'),
            'N_Conc (g/L)': n_val,
            'Mode (รูปแบบ)': data.get('mode', 'Batch'),
            'Agitation (rpm)': float(data.get('rpm', 200)),
            'Aeration': data.get('aeration', 'Anaerobic'),
            'DO / Gas Flow': float(data.get('doGas', 0)),
            'Temp (°C)': t_val,
            'Temp (┬░C)': t_val, # ป้องกันปัญหา Encoding ของสัญลักษณ์องศา
            'pH': p_val,
            'Time (h)': time_val,
            'CN_Ratio': cn_ratio,
            'C_per_Time': c_per_time,
            'Temp_pH': temp_ph,
            'Strain_Enc': strain_enc
        }

        # เรียงลำดับคอลัมน์ให้ตรงเป๊ะ 100% ตามที่โมเดลระบุไว้
        input_dict = {}
        for feat in EXPECTED_FEATURES:
            input_dict[feat] = raw_input.get(feat, 0)
            
        df_input = pd.DataFrame([input_dict])
        
        # 4. ทำนายผล
        prediction = model.predict(df_input)[0]
        
        return jsonify({
            'status': 'success',
            'yield': float(prediction),
            'derived': {
                'cn_ratio': round(cn_ratio, 2),
                'c_per_h': round(c_per_time, 2),
                'temp_ph': round(temp_ph, 2)
            }
        })

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # รัน Server
    app.run(host='0.0.0.0', port=5001, debug=False)