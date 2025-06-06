import pandas as pd
import numpy as np
import time
from scipy.io import arff
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import chebyshev
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DataClusteringApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Приложение для кластеризации данных")
        self.minsize(600, 400)
        self.geometry("750x600")

        self.original_data = pd.DataFrame()
        self.current_data = pd.DataFrame()
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side="left", fill="y", padx=5)
        control_frame.pack_propagate(False)

        ttk.Label(control_frame, text="Управление датасетом", font=("Arial", 10, "bold")).pack(pady=5)

        self.load_button = ttk.Button(control_frame, text="Загрузить датасет", command=self.load_dataset)
        self.load_button.pack(fill="x", pady=5)

        self.save_button = ttk.Button(control_frame, text="Сохранить датасет", command=self.save_dataset)
        self.save_button.pack(fill="x", pady=5)

        self.reset_button = ttk.Button(control_frame, text="Сбросить датасет", command=self.reset_dataset)
        self.reset_button.pack(fill="x", pady=5)

        self.deidentify_button = ttk.Button(control_frame, text="Обезличить датасет", command=self.deidentify_dataset)
        self.deidentify_button.pack(fill="x", pady=5)

        self.visualize_button = ttk.Button(control_frame, text="Визуализировать кластеры", command=self.visualize_clusters)
        self.visualize_button.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Число признаков\n"+"(если пусто — будут задействованы все):").pack(fill="x", pady=5)
        self.feature_count_entry = ttk.Entry(control_frame)
        self.feature_count_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Число кластеров\n"+"(если пусто — подберётся автоматически):").pack(fill="x", pady=5)
        self.cluster_count_entry = ttk.Entry(control_frame)
        self.cluster_count_entry.pack(fill="x", pady=5)
        self.cluster_button = ttk.Button(control_frame, text="Запустить кластеризацию", command=self.perform_clustering)
        self.cluster_button.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Результаты:", font=("Arial", 10, "bold")).pack(pady=5)
        self.status_text = tk.Text(control_frame, height=12, bg="#f0f0f0", state="disabled", font=("Arial", 10))
        self.status_text.pack(fill="both", expand=True, pady=15)

        table_frame = ttk.Frame(main_frame)
        table_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.data_table = ttk.Treeview(table_frame, show="headings")
        self.data_table.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_table.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.data_table.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.data_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def save_dataset(self):
        if self.current_data.empty:
            messagebox.showwarning("Ошибка", "Датасет не загружен.")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Excel файлы", "*.xlsx"), ("CSV файлы", "*.csv")],
                title="Сохранить датасет"
            )
            if file_path:
                if file_path.lower().endswith('.xlsx'):
                    self.current_data.to_excel(file_path, index=False, engine='openpyxl')
                else:
                    self.current_data.to_csv(file_path, index=False)
                self.update_status(f"Датасет успешно сохранён в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении датасета: {e}")

    def reset_dataset(self):
        if self.original_data.empty:
            messagebox.showwarning("Ошибка", "Оригинальный датасет не загружен.")
            return
        self.current_data = self.original_data.copy()
        self.update_table_display()
        self.update_status("Датасет сброшен до исходного состояния.")

    def update_table_display(self):
        for item in self.data_table.get_children():
            self.data_table.delete(item)
        if self.current_data.empty:
            return
        self.data_table["columns"] = self.current_data.columns.tolist()
        for col in self.current_data.columns:
            self.data_table.heading(col, text=col)
            max_length = max(
                len(str(col)),
                max((len(str(val)) for val in self.current_data[col]), default=10)
            )
            width = min(max_length * 8, 100)
            self.data_table.column(col, width=width, stretch=False)
        for _, row in self.current_data.iterrows():
            self.data_table.insert("", "end", values=row.tolist())

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv"), ("ARFF файлы", "*.arff")])
        if not file_path:
            return
        try:
            if file_path.lower().endswith('.arff'):
                data, meta = arff.loadarff(file_path)
                df = pd.DataFrame(data)
                for col in df.select_dtypes([object]).columns:
                    if isinstance(df[col].iloc[0], (bytes, bytearray)):
                        df[col] = df[col].str.decode('utf-8')
                self.current_data = df
            else:
                self.current_data = pd.read_csv(file_path)
            if len(self.current_data) < 2:
                messagebox.showerror("Ошибка", "Датасет должен содержать не менее 2 строк.")
                self.current_data = pd.DataFrame()
                self.data_table.delete(*self.data_table.get_children())
                return
            if len(self.current_data.columns) < 1:
                messagebox.showerror("Ошибка", "Датасет должен содержать хотя бы один признак.")
                self.current_data = pd.DataFrame()
                self.data_table.delete(*self.data_table.get_children())
                return
            for col in self.current_data.columns:
                if self.current_data[col].isna().any():
                    if pd.api.types.is_numeric_dtype(self.current_data[col]):
                        self.current_data[col] = self.current_data[col].fillna(self.current_data[col].median())
                    else:
                        self.current_data[col] = self.current_data[col].fillna(self.current_data[col].mode()[0])
            self.original_data = self.current_data.copy()
            self.update_table_display()
            self.update_status("Датасет успешно загружен.")
        except Exception as e:
            self.current_data = pd.DataFrame()
            self.data_table.delete(*self.data_table.get_children())
            self.update_status(f"Ошибка загрузки датасета: {e}")

    def update_status(self, message):
        self.status_text.config(state="normal")
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state="disabled")

    def visualize_clusters(self):
        if self.current_data.empty:
            messagebox.showwarning("Ошибка", "Датасет не загружен.")
            return
        if 'Predicted_Cluster' not in self.current_data.columns:
            messagebox.showwarning("Ошибка", "Сначала выполните кластеризацию.")
            return

        try:
            data = self.current_data.copy()
            labels = data['Predicted_Cluster'].values
            data = data.drop(columns=['Predicted_Cluster'], errors='ignore')

            for column in data.columns:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column].astype(str))

            feature_matrix = data.values
            if len(feature_matrix) < 2 or len(feature_matrix[0]) < 1:
                messagebox.showerror("Ошибка", "Недостаточно данных для визуализации.")
                return

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_matrix)

            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.xlabel('Главная компонента 1')
            plt.ylabel('Главная компонента 2')
            plt.title(f'Визуализация кластеров (PCA)\nPC1: {explained_variance[0]:.2%}, PC2: {explained_variance[1]:.2%}')
            plt.legend(*scatter.legend_elements(), title="Кластеры")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при визуализации кластеров: {e}")

    def compute_compactness(self, data, labels):
        unique_labels = np.unique(labels)
        compactness = 0
        for label in unique_labels:
            cluster_points = data[labels == label]
            if len(cluster_points) > 1:
                centroid = np.mean(cluster_points, axis=0)
                distances = np.array([chebyshev(point, centroid) for point in cluster_points])
                compactness += np.sum(distances) / len(cluster_points)
        return compactness / len(unique_labels) if unique_labels.size > 0 else 0

    def select_features_by_compactness(self, data, n_features, n_clusters):
        n_samples, n_features_total = data.shape
        if n_features_total < 1 or n_samples < 2:
            return []
        best_features = []
        remaining_indices = list(range(n_features_total))
        current_data = np.zeros((n_samples, 0))

        for _ in range(min(n_features, n_features_total)):
            best_score = float('inf')
            best_index = None
            for idx in remaining_indices:
                temp_data = np.column_stack((current_data, data[:, idx]))
                try:
                    temp_labels = linkage(temp_data, method='average', metric='chebyshev')
                    temp_labels = fcluster(temp_labels, t=n_clusters, criterion='maxclust')
                    score = self.compute_compactness(temp_data, temp_labels)
                    if score < best_score:
                        best_score = score
                        best_index = idx
                except Exception:
                    continue
            if best_index is not None:
                best_features.append(best_index)
                current_data = np.column_stack((current_data, data[:, best_index]))
                remaining_indices.remove(best_index)
        return best_features

    def compute_separability(self, data, labels):
        if len(np.unique(labels)) < 2 or len(data) < 2:
            return 0.0
        try:
            return silhouette_score(data, labels, metric='chebyshev')
        except Exception:
            return 0.0

    def find_optimal_clusters(self, data, max_clusters=10):
        best_n_clusters = 2
        best_score = -1
        linkage_matrix = linkage(data, method='average', metric='chebyshev')
        silhouette_scores = []
        for n_clusters in range(2, min(max_clusters + 1, len(data))):
            try:
                labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
                score = self.compute_separability(data, labels)
                silhouette_scores.append((n_clusters, score))
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception:
                silhouette_scores.append((n_clusters, 0.0))
        if best_score < 0.3 and best_n_clusters == 2:
            for n_clusters, score in silhouette_scores:
                if n_clusters >= 3 and score > best_score * 0.9:
                    best_n_clusters = n_clusters
                    best_score = score
                    break
        if best_score < 0.1:
            silhouette_scores.append((0, "Предупреждение: низкие силуэтные коэффициенты указывают на плохую разделимость данных.\nРекомендация: выполните кластеризацию до обезличивания или используйте более мягкие правила обезличивания (например, увеличьте число бинов)."))
        return best_n_clusters, best_score, silhouette_scores

    def perform_clustering(self):
        if self.current_data.empty:
            messagebox.showwarning("Ошибка", "Датасет не загружен.")
            return
        try:
            n_features_input = self.feature_count_entry.get().strip()
            n_features = int(n_features_input) if n_features_input else 0
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите целое число признаков или оставьте поле пустым.")
            return
        try:
            n_clusters_input = self.cluster_count_entry.get().strip()
            n_clusters = int(n_clusters_input) if n_clusters_input else 0
            if n_clusters_input and n_clusters < 2:
                raise ValueError("Число кластеров должно быть 2 или больше.")
        except ValueError as e:
            messagebox.showwarning("Ошибка", str(e) or "Введите целое число кластеров (ничего для авто, ≥2 вручную).")
            return

        total_features = len(self.current_data.columns)
        if 'Predicted_Cluster' in self.current_data.columns:
            total_features -= 1
        if n_features < 0 or n_features > total_features:
            messagebox.showwarning("Ошибка", f"Число признаков должно быть от 0 до {total_features}.")
            return

        data = self.current_data.copy()
        try:
            for column in data.columns:
                if column == 'Predicted_Cluster':
                    continue
                if column in ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                    if data[column].isna().any():
                        data[column] = data[column].fillna(data[column].median())
                else:
                    data[column] = data[column].astype(str)
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка кодирования данных: {e}")
            return

        feature_matrix = data.drop(columns=['Predicted_Cluster'], errors='ignore').values
        if len(feature_matrix) < 2 or len(feature_matrix[0]) < 1:
            messagebox.showerror("Ошибка", "Недостаточно данных для кластеризации.")
            return

        start_time = time.perf_counter()

        if n_features == 0:
            reduced_matrix = feature_matrix
            selected_features_count = total_features
        else:
            selected_indices = self.select_features_by_compactness(feature_matrix, n_features, n_clusters or 2)
            if not selected_indices:
                messagebox.showerror("Ошибка", "Не удалось выбрать признаки для кластеризации.")
                return
            reduced_matrix = feature_matrix[:, selected_indices]
            selected_features_count = len(selected_indices)

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reduced_matrix)

        try:
            linkage_matrix = linkage(normalized_features, method='average', metric='chebyshev')
            if n_clusters == 0:
                n_clusters, separability, silhouette_scores = self.find_optimal_clusters(normalized_features)
                predicted_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
                # Проверка распределения точек по кластерам
                cluster_counts = pd.Series(predicted_labels).value_counts().sort_index()
                effective_clusters = sum(cluster_counts > 1)  # Считаем кластеры с более чем 1 точкой
                separability_str = f"{separability:.4f}" if isinstance(separability, (int, float)) else "не удалось вычислить"
                status_lines = [
                    f"Отобрано признаков: {selected_features_count}",
                    f"Число кластеров: {n_clusters} (авто, эффективных: {effective_clusters})",
                    f"Отделимость кластеров: {separability_str}",
                    f"Время выполнения: {(time.perf_counter() - start_time):.2f} сек.",
                    "Силуэтные коэффициенты:"
                ]
                for item in silhouette_scores:
                    if isinstance(item, tuple):
                        n, score = item
                        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "не удалось вычислить"
                        status_lines.append(f"  {n} кластеров: {score_str}")
                    else:
                        status_lines.append(f"  {item}")
                status_lines.append("Распределение точек по кластерам:")
                for cluster, count in cluster_counts.items():
                    status_lines.append(f"  Кластер {cluster}: {count} точек")
            else:
                predicted_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
                separability = self.compute_separability(normalized_features, predicted_labels)
                cluster_counts = pd.Series(predicted_labels).value_counts().sort_index()
                effective_clusters = sum(cluster_counts > 1)
                separability_str = f"{separability:.4f}" if isinstance(separability, (int, float)) else "не удалось вычислить"
                status_lines = [
                    f"Отобрано признаков: {selected_features_count}",
                    f"Число кластеров: {n_clusters} (эффективных: {effective_clusters})",
                    f"Отделимость кластеров: {separability_str}",
                    f"Время выполнения: {(time.perf_counter() - start_time):.2f} сек.",
                    "Распределение точек по кластерам:"
                ]
                for cluster, count in cluster_counts.items():
                    status_lines.append(f"  Кластер {cluster}: {count} точек")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка кластеризации: {e}")
            return

        self.current_data['Predicted_Cluster'] = predicted_labels
        self.update_table_display()
        self.update_status("\n".join(status_lines))

    def deidentify_dataset(self):
        if self.current_data.empty:
            messagebox.showwarning("Ошибка", "Датасет не загружен.")
            return
        if len(self.current_data) < 5:
            messagebox.showwarning("Ошибка", "Датасет должен содержать не менее 5 строк для k-анонимности.")
            return

        anonymized_df = self.current_data.copy()

        numeric_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
        for col in numeric_columns:
            if col in anonymized_df.columns:
                anonymized_df[col] = pd.to_numeric(anonymized_df[col], errors='coerce')
                if anonymized_df[col].isna().any():
                    anonymized_df[col] = anonymized_df[col].fillna(anonymized_df[col].median())

        if 'Age' in anonymized_df.columns:
            bins = [0, 10, 20, 30, 40, 50, 60, 70]  
            mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            anonymized_df['Age'] = pd.cut(
                anonymized_df['Age'],
                bins=bins,
                labels=mids,
                include_lowest=True
            )
            anonymized_df['Age'] = pd.to_numeric(anonymized_df['Age'], errors='coerce')
            if anonymized_df['Age'].isna().any():
                anonymized_df['Age'] = anonymized_df['Age'].fillna(anonymized_df['Age'].median())

        if 'Purchase Amount (USD)' in anonymized_df.columns:
            bins = [0, 20, 40, 60, 80, 100, 120, 150]  
            mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            max_value = max(150, anonymized_df['Purchase Amount (USD)'].max())
            mids[-1] = max_value
            anonymized_df['Purchase Amount (USD)'] = pd.cut(
                anonymized_df['Purchase Amount (USD)'],
                bins=bins,
                labels=mids,
                include_lowest=True
            )
            anonymized_df['Purchase Amount (USD)'] = pd.to_numeric(anonymized_df['Purchase Amount (USD)'], errors='coerce')
            if anonymized_df['Purchase Amount (USD)'].isna().any():
                anonymized_df['Purchase Amount (USD)'] = anonymized_df['Purchase Amount (USD)'].fillna(anonymized_df['Purchase Amount (USD)'].median())

        if 'Review Rating' in anonymized_df.columns:
            bins = [0, 1.0, 2.0, 3.0, 4.0, 5.0]  
            mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            anonymized_df['Review Rating'] = pd.cut(
                anonymized_df['Review Rating'],
                bins=bins,
                labels=mids,
                include_lowest=True
            )
            anonymized_df['Review Rating'] = pd.to_numeric(anonymized_df['Review Rating'], errors='coerce')
            if anonymized_df['Review Rating'].isna().any():
                anonymized_df['Review Rating'] = anonymized_df['Review Rating'].fillna(anonymized_df['Review Rating'].median())

        if 'Previous Purchases' in anonymized_df.columns:
            max_purchases = anonymized_df['Previous Purchases'].max()
            bins = [-1, 0, 5, 10, 20, 30, 40, 50]  
            if max_purchases > 50:
                bins.append(max_purchases)
            else:
                bins[-1] = max_purchases + 1
            mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            anonymized_df['Previous Purchases'] = pd.cut(
                anonymized_df['Previous Purchases'],
                bins=bins,
                labels=mids,
                include_lowest=True
            )
            anonymized_df['Previous Purchases'] = pd.to_numeric(anonymized_df['Previous Purchases'], errors='coerce')
            if anonymized_df['Previous Purchases'].isna().any():
                anonymized_df['Previous Purchases'] = anonymized_df['Previous Purchases'].fillna(anonymized_df['Previous Purchases'].median())

        # if 'Category' in anonymized_df.columns:
        #     top_categories = {'Clothing', 'Footwear', 'Accessories'}  
        #     anonymized_df['Category'] = anonymized_df['Category'].astype(str).apply(
        #         lambda x: x if x in top_categories else 'Other'
        #     )
        # if 'Color' in anonymized_df.columns:
        #     top_colors = {'Gray', 'Maroon', 'White', 'Black', 'Blue'}  # Оставляем больше цветов
        #     anonymized_df['Color'] = anonymized_df['Color'].astype(str).apply(
        #         lambda x: x if x in top_colors else 'Other'
        #     )
        # if 'Size' in anonymized_df.columns:
        #     anonymized_df['Size'] = anonymized_df['Size'].astype(str).apply(
        #         lambda x: 'S/M' if x in ['S', 'M'] else 'L/XL'
        #     )
        # if 'Payment Method' in anonymized_df.columns:
        #     anonymized_df['Payment Method'] = anonymized_df['Payment Method'].astype(str).apply(
        #         lambda x: 'Digital' if x in ['Venmo', 'PayPal', 'Credit Card'] else 'Other'
        #     )
        # if 'Frequency of Purchases' in anonymized_df.columns:
        #     anonymized_df['Frequency of Purchases'] = anonymized_df['Frequency of Purchases'].astype(str).apply(
        #         lambda x: 'Frequent' if x in ['Weekly', 'Fortnightly', 'Bi-Weekly'] else 'Infrequent'
        #     )

        if 'Customer ID' in anonymized_df.columns:
            anonymized_df = anonymized_df.drop(columns=['Customer ID'])

        self.current_data = anonymized_df
        self.update_table_display()

        quasi_identifier_columns = [
            col for col in ['Age', 'Gender', 'Category', 'Purchase Amount (USD)']
            if col in anonymized_df.columns and col != 'Predicted_Cluster'
        ]
        if not quasi_identifier_columns:
            self.update_status("Невозможно вычислить k-анонимность: нет подходящих признаков.")
            return

        try:
            quasi_identifiers = anonymized_df[quasi_identifier_columns].copy()
            for col in quasi_identifiers.columns:
                if col in ['Age', 'Purchase Amount (USD)']:
                    quasi_identifiers[col] = pd.to_numeric(quasi_identifiers[col], errors='coerce')
                    if quasi_identifiers[col].isna().any():
                        quasi_identifiers[col] = quasi_identifiers[col].fillna(quasi_identifiers[col].median())
                else:
                    quasi_identifiers[col] = quasi_identifiers[col].astype(str)
                    if quasi_identifiers[col].isna().any():
                        quasi_identifiers[col] = quasi_identifiers[col].fillna(quasi_identifiers[col].mode()[0])

            status_lines = ["Диагностика перед группировкой:"]
            for col in quasi_identifiers.columns:
                status_lines.append(f"Столбец {col}: тип {quasi_identifiers[col].dtype}, пропуски {quasi_identifiers[col].isna().sum()}")

            group_counts = quasi_identifiers.groupby(quasi_identifiers.columns.tolist(), observed=True).size()
            if group_counts.empty:
                self.update_status("\n".join(status_lines + ["Невозможно вычислить k-анонимность: нет данных после группировки."]))
                return

            most_common_key = group_counts.idxmax()
            keys_per_row = quasi_identifiers.apply(lambda row: tuple(row), axis=1)
            row_group_sizes = keys_per_row.map(group_counts)
            low_size_mask = row_group_sizes < 5 
            modified_rows = low_size_mask.sum()

            if low_size_mask.any():
                replacement_df = pd.DataFrame(
                    [most_common_key] * modified_rows,
                    columns=quasi_identifiers.columns,
                    index=quasi_identifiers[low_size_mask].index
                )
                anonymized_df.loc[low_size_mask, quasi_identifiers.columns] = replacement_df
                self.current_data = anonymized_df

            new_group_counts = anonymized_df[quasi_identifier_columns].groupby(
                quasi_identifiers.columns.tolist(), observed=True
            ).size()
            k_anonymity = new_group_counts.min() if not new_group_counts.empty else 0
            size_distribution = new_group_counts.value_counts().sort_index()
            top_sizes = size_distribution.head(10)

            status_lines.extend([
                f"k-анонимность датасета: {k_anonymity}",
                f"Изменено строк для k=5: {modified_rows}",
                "Размеры групп и их количество:"
            ])
            for size, count in top_sizes.items():
                status_lines.append(f"{size}: {count}")
            self.update_status("\n".join(status_lines))
        except Exception as e:
            self.update_status(f"Ошибка при вычислении k-анонимности: {str(e)}\nТип исключения: {type(e).__name__}")

if __name__ == '__main__':
    app = DataClusteringApp()
    app.mainloop()