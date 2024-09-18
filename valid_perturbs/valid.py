import pandas as pd

# read by default 1st sheet of an excel file
PRINT_SCORE = False
DB_DATA_FILE = "all_data_25_07_23.xlsx"
EXCEL_TO_CHECK = "EXCEL_FILE_TO_CHECK"  # ONLY ENTER FILE NAME, do not enter .xlsx or other extensions

# Source of truth DB compared against from SIGNOR database
# https://signor.uniroma2.it/
signor_data = pd.read_excel(DB_DATA_FILE)[["ENTITYA", "ENTITYB", "EFFECT", "PMID"]]
allowed_down_effects = [
    "down-regulates",
    "down-regulates activity",
    "down-regulates quantity",
    "down-regulates quantity by repression",
    "down-regulates quantity by destabilization",
]
allowed_up_effects = [
    "up-regulates",
    "up-regulates activity",
    "up-regulates quantity",
    "up-regulates quantity by expression",
    "up-regulates quantity by stabilization",
]

# mapping names to exact strings in SIGNOR database, any one of these matching will trigger a match
name_map = {
    "CycD": ["CyclinD/CDK4", "CDK4", "CDK6", "CCND1", "CDK6/CCND1"],
    "CycE": ["CyclinE/CDK2", "CDK2", "CCNE1"],
    "CycA": ["CyclinA2/CDK2", "CDK2", "CCNA1"],
    "CycB": ["CyclinB/CDK1", "CDK1", "CCNB1"],
    "E2F": ["E2F1"],
    "Skp2": ["SKP2"],
    "Cdh1": ["CDH1", "FZR1"],
    "Cdc25": ["CDC25A", "CDC25C", "CDC25B"],
    "RB": ["RB1"],
    "P21": ["CDKN1B", "CDKN1A"],
    "P27": ["CDKN1B", "CDKN1A"],
    "Cdc20": ["CDC20"],
    "Wee1": ["WEE1"],
    "Cdc14": ["CDC14B", "CDC14A", "PPP1CA"],
    "Plk1": ["PLK1"],
    "Pkmyt1": ["PKMYT1"],
    "Aurka": ["AURKA"],
}

signor_data_map = {}
for i in signor_data.to_numpy():
    entity_a = str(i[0]).strip()
    entity_b = str(i[1]).strip()
    regulation = str(i[2]).strip()
    pmid = str(i[3]).strip()
    signor_data_map[(entity_a, entity_b)] = (regulation, pmid)
pmid_map = {}

"""
returns a list of (c1, c2, effect)
if double perturb, returns list of length 2
"""


def parse_perturb(perturb):
    parsed_perturbs = []

    perturbs = perturb.split("|")
    for p in perturbs:
        pair, effect = p.split("->")
        c1, c2 = pair.strip().split("-to-")
        effect = int(effect.strip().split("to")[1])  # what the pair is perturbed to, eg. 1 in 0to1
        parsed_perturbs.append((c1, c2, effect))

    return parsed_perturbs


"""
returns True if perturb data found in the database
    c1 and c2 -> one of the keys in $name_map$
    effect -> -1 if down, 1 if up, ignores 0
as soon as 1 matching pair is found in name_map values for that key, it is counted as valid
"""


def check_perturb(c1, c2, effect):
    effect_0_valid = True  # by default we assume the c1,c2 pair is not in DB
    for c1_name in name_map[c1]:
        for c2_name in name_map[c2]:
            if (c1_name, c2_name) in signor_data_map:
                # if we find a pair in signor_data_map, it means effect 0 case is not valid anymore
                if effect == 0:
                    effect_0_valid = False

                effect_in_db, pmid = signor_data_map[(c1_name, c2_name)]
                # for effect 1 or -1, we can immediately return True when found
                if (effect_in_db in allowed_up_effects and effect == 1) or (
                    effect_in_db in allowed_down_effects and effect == -1
                ):
                    item = (
                        c1_name,
                        c2_name,
                        effect,
                        pmid,
                    )
                    if (c1, c2, effect) in pmid_map:
                        pmid_map[(c1, c2, effect)].add(item)
                    else:
                        pmid_map[(c1, c2, effect)] = set()
                        pmid_map[(c1, c2, effect)].add(item)
                    return True
    return effect_0_valid if effect == 0 else False


def map_cols_to_isValid(value):
    pert = value.strip()
    if pert in perturbs_result:
        return perturbs_result[pert]["valid"]
    return ""


def map_cols_to_dbContext(value):
    pert = value.strip()
    if pert in perturbs_result:
        db_context_string = " ".join(
            map(lambda x: "[%s -> %s -> %s -> %s]" % x, perturbs_result[pert].get("db_context", []))
        )
        return db_context_string
    return ""


if __name__ == "__main__":

    model_perturb_data = pd.read_excel("scan/%s.xlsx" % EXCEL_TO_CHECK)

    # for the single perturb mode, both of the below dictionaries mean the same

    perturbs_result = {}  # holds True/False if a line item in the perturb excel sheet is good or not
    cols = ["Graph Modification ID"] + (["Graph Score"] if PRINT_SCORE else [])
    for row in model_perturb_data[cols].to_dict("records"):
        perturb = row["Graph Modification ID"]
        if PRINT_SCORE:
            score = row["Graph Score"]
        db_context = []
        # if perturb in results and results[perturb]: continue # if perturb seen before, dont process again

        if "->" not in perturb:
            print("Skipping perturb: " + perturb)
            continue

        row_result = None

        for c1, c2, effect in parse_perturb(perturb):

            perturb_result = None
            if c1 in name_map and c2 in name_map and c1 != c2:
                perturb_result = check_perturb(c1, c2, effect)
            # if the first peturb in this set was not checked, determine result using this perturb, otherwise && with first one
            if perturb_result is not None:

                # store result in row_result
                if row_result is None:
                    row_result = perturb_result
                else:
                    row_result = row_result and perturb_result

            if (c1, c2, effect) in pmid_map:
                db_context.extend(pmid_map[(c1, c2, effect)])

        if row_result is not None:
            perturbs_result[perturb] = {"valid": row_result, "score": (score if PRINT_SCORE else "")}
            if len(db_context) > 0:
                perturbs_result[perturb].update({"db_context": db_context})

    print(
        "%d out of %d perturbs checked are in DB"
        % (len(list(filter(lambda x: x["valid"], perturbs_result.values()))), len(perturbs_result.values()))
    )

    model_perturb_data["Exists in DB"] = model_perturb_data["Graph Modification ID"].apply(map_cols_to_isValid)
    model_perturb_data["DB Context"] = model_perturb_data["Graph Modification ID"].apply(map_cols_to_dbContext)

    model_perturb_data.to_excel("added_%s.xlsx" % EXCEL_TO_CHECK)
