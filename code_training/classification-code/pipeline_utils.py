import find_usefull_images_scripts as im_script
import cv2
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import seaborn as sns
## -------- Evaluation --------
def log_class_metrics_heatmap(val_results, null_classes=[], wandb_key="val/class_metrics_heatmap"):
    """
    Erstellt eine Heatmap aus den Klassenspezifischen Metriken (Precision, Recall, F1, AP@0.5)
    aus den val_results eines YOLOv8-Modells und loggt sie zu Weights & Biases.
    Diese Version ist robust gegenüber fehlenden Klassen im Val-Set.
    
    Parameter: 
        val_results: Das Ergebnisobjekt von model.val()
        wandb_key (str): Der Key unter dem das Bild bei W&B geloggt wird
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import wandb

    # Klassennamen sortiert
    names_dict = val_results.names
    sorted_class_ids_and_names = sorted(names_dict.items())
    print("sorted_class_ids_and_names ", sorted_class_ids_and_names)
    
    map_id_on_result_id = {}
    count = 0
    for i, name in sorted_class_ids_and_names:
        if name in null_classes:
            map_id_on_result_id[i] = None 
        else:
            map_id_on_result_id[i] = count
            count += 1
        
    names = [name for _, name in sorted_class_ids_and_names if name not in null_classes]
    class_ids = [i for i, _ in sorted_class_ids_and_names]

    # Zugriff auf Metriken
    p = val_results.box.p if hasattr(val_results.box, 'p') else []
    r = val_results.box.r if hasattr(val_results.box, 'r') else []
    f1 = val_results.box.f1 if hasattr(val_results.box, 'f1') else []
    ap = val_results.box.all_ap if hasattr(val_results.box, 'all_ap') else []

    # Hilfsfunktion zum sicheren Zugriff
    def safe_get(metric_list, idx, default=0.0):
        return metric_list[idx] if idx < len(metric_list) else default

    def safe_ap0(metric_list, idx):
        return metric_list[idx][0] if idx < len(metric_list) and len(metric_list[idx]) > 0 else 0.0

    # Metriken extrahieren pro Klasse
    precisions = [safe_get(p, map_id_on_result_id[i]) for i in class_ids if map_id_on_result_id[i] != None]
    recalls = [safe_get(r, map_id_on_result_id[i]) for i in class_ids if map_id_on_result_id[i] != None]
    f1s = [safe_get(f1, map_id_on_result_id[i]) for i in class_ids if map_id_on_result_id[i] != None]
    ap50s = [safe_ap0(ap, map_id_on_result_id[i]) for i in class_ids if map_id_on_result_id[i] != None]

    metrics_matrix = np.array([
        precisions,
        recalls,
        f1s,
        ap50s
    ])

    metric_names = ['Precision', 'Recall', 'F1', 'AP@0.5']

    # Heatmap erzeugen
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 4))
    im = ax.imshow(metrics_matrix, cmap='viridis', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names)

    for i in range(metrics_matrix.shape[0]):
        for j in range(metrics_matrix.shape[1]):
            ax.text(j, i, f"{metrics_matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if metrics_matrix[i, j] < 0.5 else "black")

    plt.colorbar(im, ax=ax)
    plt.title("Metriken pro Klasse")
    plt.tight_layout()

    wandb.log({wandb_key: wandb.Image(fig)})

    plt.close(fig)


def mvtec_grids(model, mvtec_f):
    image_paths, _ = mvtec_f()
    batch_size = 20
    num_grids = 10

    for grid_idx in range(num_grids):
        start_idx = grid_idx * batch_size
        end_idx = start_idx + batch_size
        selected_paths = image_paths[start_idx:end_idx]

        # Vorhersagen durchführen (Batch)
        preds = model.predict(
            selected_paths,
            imgsz=model.args["imgsz"],
            save=False,
            stream=False,
            verbose=False
        )

        # Bilder vorbereiten
        images_drawn = []
        for img_path, pred in zip(selected_paths, preds):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, _ = img.shape
            for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]
                label = f"{class_name} {conf:.2f}"

                # Rechteck zeichnen
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Textgröße bestimmen
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Textposition
                text_x = x1
                if y1 - text_h - baseline > 0:
                    text_y = y1 - 5
                    # Hintergrundrechteck für Text (oben)
                    cv2.rectangle(img, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                else:
                    text_y = y2 + text_h + 5
                    if text_y > h:
                        text_y = y2 - 5
                    # Hintergrundrechteck für Text (unten)
                    cv2.rectangle(img, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            images_drawn.append(img)

        # 5x4 Grid erstellen
        rows, cols = 5, 4
        fig, axs = plt.subplots(rows, cols, figsize=(12, 15), dpi=300)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.05)

        for i, ax in enumerate(axs.flat):
            if i < len(images_drawn):
                ax.imshow(images_drawn[i])
                ax.axis('off')
            else:
                ax.axis('off')

        # Grid als Bild speichern
        model_dir = os.path.dirname(getattr(model, 'weights', getattr(model, 'pt_path', '')))
        if not model_dir:
            model_dir = '.'  # fallback to current directory if path not found
        grid_img_path = os.path.join(model_dir, f"prediction_grid_{grid_idx+1}.jpg")
        fig.savefig(grid_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Bild bei wandb loggen
        wandb.log({f"mvtec/grids/prediction_grid{grid_idx+1}": wandb.Image(grid_img_path)})


def mvtec_metrics(model, path, big):
    # null_classes for big 
    null_classes = ["lemon", "oatmeal", "tomato sauce"]
    # for small dataset 
    if not big:
        null_classes = ["coffee", "lemon", "oatmeal", "pasta", "tomato sauce"]

    absolute_path = os.path.abspath(path)
    # Evaluation auf dem 'test' Teil des Datasets
    metrics = model.val(
        data=absolute_path,  
        split='test',              
        imgsz=model.args["imgsz"],
        #augment=True Augmentations during testing             
    )

    print(np.mean(metrics.box.p), np.mean(metrics.box.r), np.mean(metrics.box.f1))
    log_class_metrics_heatmap(metrics, null_classes=null_classes, wandb_key="mvtec/heatmap")
    wandb.log({
        "mvtec/mAP50_class_normal": float(metrics.box.map50),
        "mvtec/precision_class_normal": float(np.mean(metrics.box.p)),
        "mvtec/recall_class_normal": float(np.mean(metrics.box.r)),
        "mvtec/f1_class_normal": float(np.mean(metrics.box.f1)),
        "mvtec/mAP50-95_class_normal": float(metrics.box.map),
    })


def custom_grids(model, big):
    # Alle Beispielbilder laden
    if big:
        image_paths, _ = im_script.get_custom_10class_class_dataset()
    else:
        image_paths, _ = im_script.get_custom_small_class_dataset()

    batch_size = 20
    num_grids = 10

    for grid_idx in range(num_grids):
        start_idx = grid_idx * batch_size
        end_idx = start_idx + batch_size
        selected_paths = image_paths[start_idx:end_idx]

        # Vorhersagen durchführen (Batch)
        preds = model.predict(
            selected_paths,
            imgsz=model.args["imgsz"],
            save=False,
            stream=False,
            verbose=False
        )

        # Bilder vorbereiten
        images_drawn = []
        for img_path, pred in zip(selected_paths, preds):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, _ = img.shape
            for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
                x1, y1, x2, y2 = map(int, box)

                class_name = model.names[int(cls)]
                #print("mapping" , class_name, cls)
                label = f"{class_name} {conf:.2f}"

                # Rechteck zeichnen
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Textgröße bestimmen
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Textposition
                text_x = x1
                if y1 - text_h - baseline > 0:
                    text_y = y1 - 5
                    # Hintergrundrechteck für Text (oben)
                    cv2.rectangle(img, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                else:
                    text_y = y2 + text_h + 5
                    if text_y > h:
                        text_y = y2 - 5
                    # Hintergrundrechteck für Text (unten)
                    cv2.rectangle(img, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            images_drawn.append(img)

        # 5x4 Grid erstellen
        rows, cols = 5, 4
        fig, axs = plt.subplots(rows, cols, figsize=(12, 15), dpi=300)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.05)

        for i, ax in enumerate(axs.flat):
            if i < len(images_drawn):
                ax.imshow(images_drawn[i])
                ax.axis('off')
            else:
                ax.axis('off')

        # Grid als Bild speichern
        # Extract model path and save grid image there
        model_dir = os.path.dirname(getattr(model, 'weights', getattr(model, 'pt_path', '')))
        if not model_dir:
            model_dir = '.'  # fallback to current directory if path not found
        grid_img_path = os.path.join(model_dir, f"prediction_grid_{grid_idx+1}.jpg")
        fig.savefig(grid_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Bild bei wandb loggen
        wandb.log({f"custom/grids/prediction_grid_{grid_idx+1}": wandb.Image(grid_img_path)})


def custom_metrics(model, big):
    print("Calculating Custom Metrics")
    # # Mapping von Model-Output-Klasse → GT-Klasse
    label_translation_trained_on_10classes = {
        0: 1, 1: 3, 2: 4, 3: 13, 4: 48, 5: 26, 6: 2, 7: 42, 8: 9, 9: 5
    }
    if not big:
        label_translation_trained_on_10classes = {
            0: 1, 1: 3, 2: 4, 3: 48, 4: 26, 5: 2, 6: 5
        }

    def compute_classnorm_metrics(gt_dicts, pred_dicts):
        """
        Berechnet class-normalisierte Precision, Recall, F1,
        wobei Klassen ohne Vorkommen ignoriert werden.
        """
        all_classes = sorted(set().union(*[d.keys() for d in gt_dicts + pred_dicts]))
        classwise_precisions = []
        classwise_recalls = []
        classwise_f1s = []
        classwise_gt_count = []
        classwise_pred_count = []
        classwise_fp = []
        classwise_fn = []
        classwise_tp = []

        for cls in all_classes:
            tp, fp, fn = 0, 0, 0
            gt_count_class = 0
            pred_count_class = 0
            for gt, pred in zip(gt_dicts, pred_dicts):
                gt_count = gt.get(cls, 0)
                pred_count = pred.get(cls, 0)
                gt_count_class += gt_count
                pred_count_class += pred_count

                tp += min(gt_count, pred_count)
                fp += max(0, pred_count - gt_count)
                fn += max(0, gt_count - pred_count)


            # Skip class if both gt and pred are zero
            if cls == 3:
                print("avocado", tp, fp, fn)
                pass
            if (tp + fp + fn) == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            classwise_gt_count.append(gt_count_class)
            classwise_pred_count.append(pred_count_class)
            classwise_fp.append(fp)
            classwise_fn.append(fn)
            classwise_tp.append(tp)
            classwise_precisions.append(precision)
            classwise_recalls.append(recall)
            classwise_f1s.append(f1)
        
        mean_precision = np.mean(classwise_precisions) if len(classwise_precisions) > 0 else 0.0
        mean_recall = np.mean(classwise_recalls) if len(classwise_recalls) > 0 else 0.0
        mean_f1 = np.mean(classwise_f1s) if len(classwise_f1s) > 0 else 0.0

        results = {"class_norm_precision": mean_precision, "class_norm_recall": mean_recall, "class_norm_f1": mean_f1, 
                "classwise_gt_count": classwise_gt_count, "classwise_pred_count": classwise_pred_count, "classwise_fp": classwise_fp, "classwise_fn": classwise_fn,
                "classwise_tp": classwise_tp, "classwise_precisions": classwise_precisions, "classwise_recalls": classwise_recalls, "classwise_f1s" :classwise_f1s }

        return results



    def compute_global_metrics(gt_dicts, pred_dicts):
        all_classes = sorted(set().union(*[d.keys() for d in gt_dicts + pred_dicts]))

        def dict_to_vec(d, classes):
            return np.array([d.get(c, 0) for c in classes], dtype=np.float32)

        gt_arr = np.stack([dict_to_vec(d, all_classes) for d in gt_dicts])
        pred_arr = np.stack([dict_to_vec(d, all_classes) for d in pred_dicts])
        print(gt_arr, pred_arr)

        tp = np.minimum(gt_arr, pred_arr).sum()
        fp = np.maximum(pred_arr - gt_arr, 0).sum()
        fn = np.maximum(gt_arr - pred_arr, 0).sum()


        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1


    def translate_prediction_counts(pred_classes, translation_dict):
        """
        Zählt vorhergesagte Klassen und übersetzt sie in Zielklassen.
        Gibt dict {gt_class_id: count} zurück.
        """
        pred_counts = {}
        for c in pred_classes:
            mapped = translation_dict[c]
            if mapped is not None:
                # get is a cool trick standard value of 0 allows to access even though its not initialised
                pred_counts[mapped] = pred_counts.get(mapped, 0) + 1
        return pred_counts

    # === Main ===

    image_paths, label_lines = im_script.get_custom_10class_class_dataset()
    batch_size = 20

    all_gt_counts = []
    all_pred_counts = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = label_lines[i:i + batch_size]  # dicts: class_id -> count (GT-Klassen)

        # GT-Labels direkt übernehmen
        all_gt_counts.extend(batch_labels)

        # Model Predictions holen
        preds_raw = model.predict(batch_paths, imgsz=model.args["imgsz"], stream=False, verbose=False)

        for i, pred in enumerate(preds_raw):
            pred_classes = pred.boxes.cls.cpu().tolist()
            translated_pred = translate_prediction_counts(pred_classes, label_translation_trained_on_10classes)
            all_pred_counts.append(translated_pred)


    precision, recall, f1 = compute_global_metrics(all_gt_counts, all_pred_counts)


    # Zusätzlich: class-normalisierte Metriken berechnen
    results = compute_classnorm_metrics(all_gt_counts, all_pred_counts)



    print(precision, recall, f1, results["class_norm_precision"], results["class_norm_recall"], results["class_norm_f1"])
    wandb.log({
        "custom/test/precision_counts": float(precision),
        "custom/test/recall_counts": float(recall),
        "custom/test/f1_score_counts": float(f1),
        "custom/test/precision_classnorm_CARE": float(results["class_norm_precision"]),
        "custom/test/recall_classnorm_CARE": float( results["class_norm_recall"]),
        "custom/test/f1_classnorm_CARE": float( results["class_norm_f1"])
    }, step=wandb.run.step)

    custom_heatmap(model, label_translation_trained_on_10classes, results)


def custom_heatmap(model, label_translation_trained_on_10classes, results):
    print("Creating Custom Heatmap")
    # === Labels vorbereiten ===
    model_names = model.names  # z. B. {0: "apple", 1: "banana", ...}
    translation = label_translation_trained_on_10classes

    translated_class_labels = {
        gt_id: model_names[pred_id]
        for pred_id, gt_id in translation.items()
    }

    # === Werte aus results extrahieren ===
    class_ids = sorted(translated_class_labels.keys())  # Nur GT-Klassen, die in der Übersetzung vorkommen
    prec_list = []
    recall_list = []
    f1_list = []
    for idx, c in enumerate(class_ids):
        if idx < len(results["classwise_precisions"]):
            prec_list.append(results["classwise_precisions"][idx])
            recall_list.append(results["classwise_recalls"][idx])
            f1_list.append(results["classwise_f1s"][idx])
        else:
            prec_list.append(0.0)
            recall_list.append(0.0)
            f1_list.append(0.0)

    # === Heatmap zeichnen ===
    metric_matrix = np.array([prec_list, recall_list, f1_list])
    metric_labels = ["Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(max(8, len(class_ids)), 4))
    sns.heatmap(
        metric_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[translated_class_labels.get(c, str(c)) for c in class_ids],
        yticklabels=metric_labels,
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title("Per-Class Precision / Recall / F1")
    plt.xlabel("Klasse")
    plt.ylabel("Metrik")
    plt.tight_layout()

    wandb.log({"custom/per_class_metrics_heatmap": wandb.Image(fig)})
    plt.close(fig)
