import os
import ast
prices = {}
data_to_path = "./Dataset"
id_file_path = os.path.join(data_to_path, "id.txt")
prices_file_path = os.path.join(data_to_path, "prices.txt")

with open(prices_file_path, "r") as f:
    for line in f:
        # Extrahiere das Dictionary aus der Zeile
        if ": " in line:
            _, dict_str = line.strip().split(": ", 1)
            try:
                current_dict = ast.literal_eval(dict_str)
                for key, value in current_dict.items():
                    if key in prices:
                        # Überprüfen, ob Werte übereinstimmen
                        if prices[key] != value:
                            print(f"Warnung: Konflikt für Schlüssel {key}: "
                                  f"{prices[key]} vs {value}")
                    else:
                        # Schlüssel hinzufügen
                        prices[key] = value
            except Exception as e:
                print(f"Fehler beim Verarbeiten der Zeile: {line}\n{e}")

print("Kombiniertes Dictionary:")
print(prices)
