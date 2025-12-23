import os
import shutil

def organize_project():
    # 自动获取脚本所在的绝对路径（即项目根目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')
    
    # 打印一下路径，方便调试
    print(f"项目根目录: {base_dir}")
    print(f"源文件目录: {src_dir}")

    if not os.path.exists(src_dir):
        print(f"错误: 找不到 src 目录，请确保脚本放在 EE5046_Projects 文件夹下。")
        return

    # 定义目标目录结构
    structure = {
        'src/common': ['Config.py'],
        'src/task1_ecg_analysis/data': ['DataManager.py', 'ECGDataset.py'],
        'src/task1_ecg_analysis/models': ['ECGAnalyzer.py'],
        'src/task1_ecg_analysis/training': ['TrainModel.py', 'TrainProcess_new.py', 'TrainProcess.py'],
        'src/task1_ecg_analysis/visualization': ['Plot_Test.py'],
        'src/task2_multimodal_llm/data': ['MultimodelDataset.py', 'Generate_Instruction_Dataset.py'],
        'src/task2_multimodal_llm/models': ['ECGencoder.py', 'MultimodelLLM.py', 'Projector.py'],
        'src/task2_multimodal_llm/training': ['Train_Multimodel.py'],
        'src/task2_multimodal_llm/evaluation': ['Evaluate_LLM.py', 'Evaluate_LLM_new.py'],
        'src/task2_multimodal_llm/visualization': ['Plot_JsonFile.py']
    }

    for folder_rel, files in structure.items():
        # 获取文件夹的绝对路径
        target_folder = os.path.join(base_dir, folder_rel)
        
        # 1. 创建目标文件夹
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            # 创建 __init__.py
            with open(os.path.join(target_folder, '__init__.py'), 'w') as f:
                pass
            print(f"创建目录: {folder_rel}")

        # 2. 移动文件
        for file_name in files:
            # 在 src 根目录查找文件
            old_path = os.path.join(src_dir, file_name)
            new_path = os.path.join(target_folder, file_name)

            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                print(f"已移动: {file_name} -> {folder_rel}")
            # 如果文件已经被移动过（比如在目标位置了），就不再报错
            elif os.path.exists(new_path):
                pass 
            else:
                print(f"未找到文件: {file_name} (请检查是否已手动移动或名称有误)")

    print("\n整理完成！")

if __name__ == "__main__":
    organize_project()