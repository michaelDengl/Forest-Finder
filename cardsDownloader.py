# cardsDownloader_stream.py
import requests, ijson

OUT_DE = "all_cards_de.txt"
OUT_EN = "all_cards_en.txt"

# 1) Get streaming URL of the bulk file
bulk = requests.get("https://api.scryfall.com/bulk-data", timeout=30).json()["data"]
url = next(x["download_uri"] for x in bulk if x["type"] in ("default_cards", "all_cards"))
print("Streaming from:", url)

# 2) Stream & parse without loading into memory
r = requests.get(url, stream=True, timeout=120)
r.raise_for_status()
r.raw.decode_content = True

seen_de, seen_en = set(), set()
cnt_de = cnt_en = 0

with open(OUT_DE, "w", encoding="utf-8") as fde, open(OUT_EN, "w", encoding="utf-8") as fen:
    for card in ijson.items(r.raw, "item"):
        lang = card.get("lang")
        if lang == "de":
            # Prefer printed_name for foreign (German) cards
            name = card.get("printed_name") or card.get("name")
            if name and name not in seen_de:
                fde.write(name + "\n")
                seen_de.add(name)
                cnt_de += 1
        elif lang == "en":
            # English oracle names are in 'name'
            name = card.get("name")
            if name and name not in seen_en:
                fen.write(name + "\n")
                seen_en.add(name)
                cnt_en += 1

print(f"Done. Wrote {cnt_de} German names to {OUT_DE}, {cnt_en} English names to {OUT_EN}.")
