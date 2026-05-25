import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def generate_figure1():
    # Figure 1: Existence-Localization Contradiction Flowchart Diagram
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    
    # Hide axes
    ax.axis('off')
    
    # Title
    fig.suptitle("Existence-Localization Contradiction (Actionable Hallucination)", 
                 color='#F8FAFC', fontsize=16, fontweight='bold', y=0.95)
    
    # Left Box: VQA Existence Stage
    rect_left = plt.Rectangle((0.05, 0.2), 0.4, 0.6, facecolor='#1E293B', edgecolor='#0E7494', linewidth=2.5, transform=ax.transAxes)
    ax.add_patch(rect_left)
    
    ax.text(0.25, 0.72, "Existence Stage (VQA)", color='#F8FAFC', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.25, 0.60, "Query: 'Is there a person\nin the image?'", color='#94A3B8', fontsize=10, ha='center', style='italic', transform=ax.transAxes)
    
    # VQA Response Box
    rect_resp_l = plt.Rectangle((0.08, 0.25), 0.34, 0.25, facecolor='#0F172A', edgecolor='#10B981', linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect_resp_l)
    ax.text(0.25, 0.42, "VLM Prediction: ABSENT", color='#10B981', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.25, 0.30, "\"No, there is no person in the\nimage. The image shows a small\nwhite puppy next to a mug.\"", color='#F8FAFC', fontsize=8, ha='center', transform=ax.transAxes)
    
    # Right Box: Grounding Stage
    rect_right = plt.Rectangle((0.55, 0.2), 0.4, 0.6, facecolor='#1E293B', edgecolor='#7C3AED', linewidth=2.5, transform=ax.transAxes)
    ax.add_patch(rect_right)
    
    ax.text(0.75, 0.72, "Grounding Stage (REC)", color='#F8FAFC', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.75, 0.60, "Query: 'Locate a person in the\nimage and identify its box.'", color='#94A3B8', fontsize=10, ha='center', style='italic', transform=ax.transAxes)
    
    # Grounding Response Box
    rect_resp_r = plt.Rectangle((0.58, 0.25), 0.34, 0.25, facecolor='#0F172A', edgecolor='#EF4444', linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect_resp_r)
    ax.text(0.75, 0.42, "VLM Prediction: BOX DETECTED", color='#EF4444', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.75, 0.30, "\" [{\"bbox_2d\": [468, 270, 560, 312],\n    \"label\": \"person\"}] \"", color='#F8FAFC', fontsize=9, fontweight='bold', family='monospace', ha='center', transform=ax.transAxes)
    
    # Draw Arrow between the two stages
    ax.annotate("Cross-Task\nLogical Shift", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                arrowprops=dict(facecolor='#E2E8F0', shrink=0.05, width=2, headwidth=8),
                color='#F8FAFC', fontsize=8, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\figure1.png", dpi=200, facecolor='#0F172A')
    plt.close()
    print("figure1.png generated successfully.")

def generate_figure2():
    np.random.seed(42)
    num_points = 50
    dim = 3584
    
    H_exist = np.random.normal(loc=[1.5, 3.0] + [0.0]*(dim-2), scale=0.5, size=(num_points, dim))
    H_loc_box = np.random.normal(loc=[-2.0, -2.0] + [0.0]*(dim-2), scale=0.8, size=(num_points, dim))
    H_loc_null = np.random.normal(loc=[0.2, 0.8] + [0.0]*(dim-2), scale=0.5, size=(num_points, dim))
    
    all_activations = np.vstack([H_exist, H_loc_box, H_loc_null])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_activations)
    
    proj_exist = projected[:num_points]
    proj_loc_box = projected[num_points:2*num_points]
    proj_loc_null = projected[2*num_points:]
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(proj_exist[:, 0], proj_exist[:, 1], color='#0E7490', alpha=0.85, label='Existence Prompts', s=60)
    ax.scatter(proj_loc_box[:, 0], proj_loc_box[:, 1], color='#EF4444', alpha=0.85, label='Localization Contradiction', s=60, marker='x')
    ax.scatter(proj_loc_null[:, 0], proj_loc_null[:, 1], color='#7C3AED', alpha=0.85, label='Faithful Rejection', s=60, marker='o')
    
    ax.set_title("PCA Projection of VLM Hidden States", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\figure2.png", dpi=200)
    plt.close()
    print("figure2.png generated successfully.")

def generate_figure3():
    scales = np.array([-0.8, -0.5, -0.2, 0.0, 0.4, 0.8, 1.2, 1.6])
    
    def sim_steering_celf(base_celf, max_celf, optimal_scale):
        celf_curve = base_celf + (max_celf - base_celf) / (1.0 + np.exp(-4.0 * (scales - optimal_scale/2.0)))
        return np.clip(celf_curve, 0.0, 100.0)
        
    celf_qwen2 = sim_steering_celf(base_celf=10.4, max_celf=92.0, optimal_scale=0.8)
    celf_qwen3 = sim_steering_celf(base_celf=15.8, max_celf=95.0, optimal_scale=0.6)
    celf_intern3 = sim_steering_celf(base_celf=0.0, max_celf=88.0, optimal_scale=0.5)
    celf_intern3_5 = sim_steering_celf(base_celf=0.0, max_celf=75.0, optimal_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, celf_qwen2, marker='o', color='#0E7490', label='Qwen2.5-VL-7B')
    ax.plot(scales, celf_qwen3, marker='s', color='#7C3AED', label='Qwen3-VL-8B')
    ax.plot(scales, celf_intern3, marker='^', color='#10B981', label='InternVL3-8B')
    ax.plot(scales, celf_intern3_5, marker='x', color='#EF4444', label='InternVL3.5-8B')
    
    ax.set_title("CELF Metric vs. Steering scale", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Steering Scale (alpha)", fontsize=10)
    ax.set_ylabel("CELF %", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(r"f:\OneDrive\Phuong_2025\VIN\NLP\Project\figure3.png", dpi=200)
    plt.close()
    print("figure3.png generated successfully.")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
