import joblib
import xgboost as xgb
import os

def inspect_xgboost_model(model_path: str):
    print("="*60)
    print(f"🔍 ĐANG KIỂM TRA MODEL: {model_path}")
    print("="*60)

    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY FILE: {model_path}")
        return

    try:
        # Load data từ file pkl
        data = joblib.load(model_path)
        booster = data.get("booster")
        feature_order = data.get("features")

        if booster is None or feature_order is None:
            print("❌ File pkl không chứa cấu trúc chuẩn {'booster': model, 'features': list_features}")
            return

        # 1. KIỂM TRA SỐ LƯỢNG VÀ THỨ TỰ FEATURE
        print(f"\n✅ 1. TỔNG SỐ FEATURES YÊU CẦU: {len(feature_order)}")
        print("-" * 40)
        for i, f in enumerate(feature_order):
            print(f"  [{i:02d}] - {f}")

        # 2. KIỂM TRA ĐỘ QUAN TRỌNG CỦA TỪNG FEATURE (FEATURE IMPORTANCE)
        # Dùng 'gain' (mức độ cải thiện độ chính xác khi dùng biến này để chia nhánh)
        try:
            importance_gain = booster.get_score(importance_type='gain')
            importance_weight = booster.get_score(importance_type='weight')
            
            print(f"\n✅ 2. BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG CỦA FEATURES (TOP GAIN)")
            print("-" * 60)
            print(f"{'TÊN FEATURE':<25} | {'ĐIỂM GAIN (Chất lượng)':<22} | {'SỐ LẦN DÙNG (Weight)'}")
            print("-" * 60)

            # Sắp xếp theo điểm Gain giảm dần
            sorted_gain = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)
            
            for f_name, gain_score in sorted_gain:
                weight_score = importance_weight.get(f_name, 0)
                print(f"{f_name:<25} | {gain_score:<22.4f} | {weight_score}")

        except Exception as e:
            print(f"⚠️ Không thể trích xuất Feature Importance: {e}")

    except Exception as e:
        print(f"❌ Lỗi khi load file: {e}")

    print("\n" + "="*60)
    print("🎯 ACTION PLAN (CÁCH DÙNG KẾT QUẢ NÀY TRONG PIPELINE)")
    print("="*60)
    print("1. Mở file tính toán features của bạn (src.core.features).")
    print("2. Chạy thử hàm compute_features_advanced() với 1 frame bất kỳ.")
    print("3. In cái dictionary trả về (feat_dict.keys()) ra console.")
    print("4. SO SÁNH: Danh sách in ra ở bước 3 CÓ KHỚP 100% (từng chữ cái) với danh sách [00] đến [14] ở trên không?")
    print("5. NẾU SAI TÊN HOẶC THIẾU: XGBoost sẽ phán đoán sai bét. Phải fix lại key trong dictionary ngay!")

if __name__ == "__main__":
    # Thay tên file model của bạn vào đây
    MODEL_FILE = "violence_model (1).pkl" 
    inspect_xgboost_model(MODEL_FILE)