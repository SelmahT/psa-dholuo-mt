import json
import itertools
import random
import pandas as pd

random.seed(42)

with open("data/knowledge_base/kenya_facts.json") as f:
    kb = json.load(f)

# ---------- Single-fact wrapper templates ----------
# Each takes one fact and presents it in a different natural PSA phrasing.
single_wrappers = [
    ("The public is informed about {fact}.", "Umma unafahamishwa kuhusu {fact_sw}."),
    ("Residents are encouraged to learn more about {fact}.", "Wakazi wanahamasishwa kujifunza zaidi kuhusu {fact_sw}."),
    ("For more information, visit your nearest county office regarding {fact}.", "Kwa maelezo zaidi, tembelea ofisi ya kaunti ya karibu kuhusu {fact_sw}."),
    ("Take advantage of {fact} available to eligible residents.", "Nufaika na {fact_sw} inayopatikana kwa wakazi wanaostahili."),
    ("Officials remind the public about {fact}.", "Maafisa wanakumbusha umma kuhusu {fact_sw}."),
    ("Eligible members of the public should follow up on {fact}.", "Wanachama wa umma wanaostahili wanapaswa kufuatilia {fact_sw}."),
    ("Don't miss out — {fact} is now open to the public.", "Usikose — {fact_sw} sasa iko wazi kwa umma."),
    ("This week's public notice concerns {fact}.", "Tangazo la wiki hii linahusu {fact_sw}."),
    ("Anyone affected should reach out regarding {fact}.", "Yeyote aliyeathirika anapaswa kuwasiliana kuhusu {fact_sw}."),
    ("The relevant authority has issued an update on {fact}.", "Mamlaka husika imetoa taarifa kuhusu {fact_sw}."),
    ("Citizens are urged to participate in {fact}.", "Wananchi wanahimizwa kushiriki katika {fact_sw}."),
    ("It is important for households to be aware of {fact}.", "Ni muhimu kwa kaya kufahamu {fact_sw}."),
    ("A new update has been issued concerning {fact}.", "Taarifa mpya imetolewa kuhusu {fact_sw}."),
    ("Community members are advised to check on {fact}.", "Wanajamii wanashauriwa kuangalia {fact_sw}."),
    ("Attention: {fact} is now available in your area.", "Angalizo: {fact_sw} sasa inapatikana katika eneo lako."),
    ("Local offices can provide guidance on {fact}.", "Ofisi za eneo zinaweza kutoa mwongozo kuhusu {fact_sw}."),
    ("Make sure your household benefits from {fact}.", "Hakikisha kaya yako inanufaika na {fact_sw}."),
    ("The government continues to expand {fact}.", "Serikali inaendelea kupanua {fact_sw}."),
    ("Please share this information about {fact} with your neighbours.", "Tafadhali shiriki taarifa hii kuhusu {fact_sw} na majirani zako."),
    ("A reminder has been sent out regarding {fact}.", "Ukumbusho umetumwa kuhusu {fact_sw}."),
    ("You may qualify for {fact} — check with local officials.", "Unaweza kustahili {fact_sw} — angalia na maafisa wa eneo lako."),
    ("Stay informed on the latest details of {fact}.", "Endelea kufahamishwa kuhusu maelezo mapya ya {fact_sw}."),
    ("Public health and civic offices are coordinating on {fact}.", "Ofisi za afya ya umma na uraia zinashirikiana kuhusu {fact_sw}."),
    ("This PSA highlights {fact} for residents to note.", "Tangazo hili linaangazia {fact_sw} kwa wakazi kuzingatia."),
    ("Follow official channels for updates on {fact}.", "Fuata njia rasmi kwa taarifa kuhusu {fact_sw}."),
    ("The relevant government department has released details about {fact}.", "Idara husika ya serikali imetoa maelezo kuhusu {fact_sw}."),
    ("Families are encouraged to inquire about {fact} at their local office.", "Familia zinahimizwa kuuliza kuhusu {fact_sw} katika ofisi yao ya eneo."),
    ("More details on {fact} can be found at your nearest public office.", "Maelezo zaidi kuhusu {fact_sw} yanapatikana katika ofisi ya umma ya karibu."),
    ("The public is reminded that {fact} remains available this quarter.", "Umma unakumbushwa kuwa {fact_sw} inaendelea kupatikana robo hii."),
    ("Residents who have not yet accessed {fact} are encouraged to do so.", "Wakazi ambao bado hawajapata {fact_sw} wanahimizwa kufanya hivyo."),
    ("An awareness drive is underway to promote {fact}.", "Kampeni ya uhamasishaji inaendelea kukuza {fact_sw}."),
    ("Local leaders are working with residents on {fact}.", "Viongozi wa eneo wanashirikiana na wakazi kuhusu {fact_sw}."),
    ("The ministry has confirmed continued support for {fact}.", "Wizara imethibitisha uungwaji mkono unaoendelea wa {fact_sw}."),
    ("Public offices remain open to assist residents with {fact}.", "Ofisi za umma zinaendelea kuwa wazi kusaidia wakazi na {fact_sw}."),
    ("This announcement serves as a reminder about {fact}.", "Tangazo hili ni ukumbusho kuhusu {fact_sw}."),
    ("Residents in need of support are directed toward {fact}.", "Wakazi wanaohitaji msaada wanaelekezwa kwenye {fact_sw}."),
    ("Local administrators continue to raise awareness on {fact}.", "Wasimamizi wa eneo wanaendelea kuhamasisha kuhusu {fact_sw}."),
    ("A public briefing has been scheduled to discuss {fact}.", "Kikao cha umma kimepangwa kujadili {fact_sw}."),
    ("Officials have called on the public to make use of {fact}.", "Maafisa wamewaomba umma kutumia {fact_sw}."),
    ("Community outreach teams are sharing information about {fact}.", "Timu za uhamasishaji wa jamii zinashiriki taarifa kuhusu {fact_sw}."),
    ("The public notice board has been updated with information on {fact}.", "Ubao wa matangazo ya umma umesasishwa na taarifa kuhusu {fact_sw}."),
    ("Local radio stations have been asked to air updates on {fact}.", "Vituo vya redio vya eneo vimeombwa kutangaza taarifa kuhusu {fact_sw}."),
    ("Residents can call their local office for help accessing {fact}.", "Wakazi wanaweza kupigia ofisi yao ya eneo kwa msaada wa kupata {fact_sw}."),
    ("An official statement has been issued regarding {fact}.", "Taarifa rasmi imetolewa kuhusu {fact_sw}."),
    ("The county administration is keen to promote {fact}.", "Utawala wa kaunti unahamasisha {fact_sw}."),
    ("This notice is issued in the public interest concerning {fact}.", "Tangazo hili linatolewa kwa maslahi ya umma kuhusu {fact_sw}."),
    ("Local chiefs have been briefed to relay information on {fact}.", "Machifu wa eneo wamepewa maelezo ya kusambaza kuhusu {fact_sw}."),
    ("Households are advised to keep updated on {fact}.", "Kaya zinashauriwa kuendelea kufahamishwa kuhusu {fact_sw}."),
    ("Public participation is welcomed regarding {fact}.", "Ushiriki wa umma unakaribishwa kuhusu {fact_sw}."),
]

# ---------- Paired-fact wrapper templates ----------
# Combines two related facts from the same domain into one compound PSA.
paired_wrappers = [
    ("In addition to {fact}, residents can also benefit from {fact2}.",
     "Mbali na {fact_sw}, wakazi wanaweza pia kunufaika na {fact2_sw}."),
    ("Alongside {fact}, the public is reminded about {fact2}.",
     "Pamoja na {fact_sw}, umma unakumbushwa kuhusu {fact2_sw}."),
    ("Both {fact} and {fact2} are available this month for eligible residents.",
     "Zote {fact_sw} na {fact2_sw} zinapatikana mwezi huu kwa wakazi wanaostahili."),
    ("Officials are coordinating {fact} together with {fact2}.",
     "Maafisa wanaratibu {fact_sw} pamoja na {fact2_sw}."),
    ("Community members can access {fact} as well as {fact2} through local offices.",
     "Wanajamii wanaweza kupata {fact_sw} pamoja na {fact2_sw} kupitia ofisi za eneo."),
    ("This week's outreach covers {fact} and {fact2}.",
     "Mkakati wa wiki hii unahusisha {fact_sw} na {fact2_sw}."),
    ("Residents seeking support can look into {fact} or {fact2}.",
     "Wakazi wanaotafuta usaidizi wanaweza kuangalia {fact_sw} au {fact2_sw}."),
    ("The county is running parallel initiatives: {fact} and {fact2}.",
     "Kaunti inaendesha mipango sambamba: {fact_sw} na {fact2_sw}."),
    ("Households can benefit from either {fact} or {fact2}, depending on eligibility.",
     "Kaya zinaweza kunufaika na {fact_sw} au {fact2_sw}, kulingana na sifa."),
    ("A joint outreach programme covers both {fact} and {fact2} this season.",
     "Mpango wa pamoja wa uhamasishaji unahusisha {fact_sw} na {fact2_sw} msimu huu."),
    ("Officials are urging the public to consider {fact} alongside {fact2}.",
     "Maafisa wanahimiza umma kuzingatia {fact_sw} pamoja na {fact2_sw}."),
    ("Residents attending local meetings will hear updates on {fact} and {fact2}.",
     "Wakazi wanaohudhuria mikutano ya eneo watasikia taarifa kuhusu {fact_sw} na {fact2_sw}."),
    ("For a complete picture, residents should review both {fact} and {fact2}.",
     "Ili kupata picha kamili, wakazi wanapaswa kuangalia {fact_sw} na {fact2_sw}."),
]

# ---------- Triple-fact wrapper templates ----------
# Combines three related facts from the same domain into one compound PSA.
triple_wrappers = [
    ("This month's public update covers {fact}, {fact2}, and {fact3}.",
     "Taarifa ya umma ya mwezi huu inahusisha {fact_sw}, {fact2_sw}, na {fact3_sw}."),
    ("Residents can access three key services: {fact}, {fact2}, and {fact3}.",
     "Wakazi wanaweza kupata huduma tatu muhimu: {fact_sw}, {fact2_sw}, na {fact3_sw}."),
    ("Local offices are coordinating efforts on {fact}, {fact2}, and {fact3}.",
     "Ofisi za eneo zinaratibu juhudi kuhusu {fact_sw}, {fact2_sw}, na {fact3_sw}."),
    ("The community outreach agenda includes {fact}, {fact2}, and {fact3}.",
     "Ajenda ya uhamasishaji wa jamii inajumuisha {fact_sw}, {fact2_sw}, na {fact3_sw}."),
    ("Households are reminded of three ongoing programmes: {fact}, {fact2}, and {fact3}.",
     "Kaya zinakumbushwa kuhusu mipango mitatu inayoendelea: {fact_sw}, {fact2_sw}, na {fact3_sw}."),
    ("This quarter's briefing highlights {fact}, {fact2}, and {fact3}.",
     "Kikao cha robo hii kinaangazia {fact_sw}, {fact2_sw}, na {fact3_sw}."),
]

rows = []
psa_counter = 1

for domain, facts in kb.items():
    # single-fact combinations
    for wrapper_en, wrapper_sw in single_wrappers:
        for f in facts:
            eng = wrapper_en.format(fact=f["fact"])
            sw = wrapper_sw.format(fact_sw=f["fact_sw"])
            rows.append({
                "PSA_Id": None, "Domain": domain, "Class": "General",
                "English": eng, "Kiswahili": sw, "Dholuo": None,
                "Fact_Source": f["id"], "Source": "grounded_generated",
            })

    # paired-fact combinations (only pairs within the same domain, no repeats)
    for wrapper_en, wrapper_sw in paired_wrappers:
        for f1, f2 in itertools.combinations(facts, 2):
            eng = wrapper_en.format(fact=f1["fact"], fact2=f2["fact"])
            sw = wrapper_sw.format(fact_sw=f1["fact_sw"], fact2_sw=f2["fact_sw"])
            rows.append({
                "PSA_Id": None, "Domain": domain, "Class": "General",
                "English": eng, "Kiswahili": sw, "Dholuo": None,
                "Fact_Source": f"{f1['id']}+{f2['id']}", "Source": "grounded_generated",
            })

    # triple-fact combinations
    for wrapper_en, wrapper_sw in triple_wrappers:
        for f1, f2, f3 in itertools.combinations(facts, 3):
            eng = wrapper_en.format(fact=f1["fact"], fact2=f2["fact"], fact3=f3["fact"])
            sw = wrapper_sw.format(fact_sw=f1["fact_sw"], fact2_sw=f2["fact_sw"], fact3_sw=f3["fact_sw"])
            rows.append({
                "PSA_Id": None, "Domain": domain, "Class": "General",
                "English": eng, "Kiswahili": sw, "Dholuo": None,
                "Fact_Source": f"{f1['id']}+{f2['id']}+{f3['id']}", "Source": "grounded_generated",
            })

df = pd.DataFrame(rows)
before = len(df)
df = df.drop_duplicates(subset="English").reset_index(drop=True)
print(f"Generated {before} rows, {len(df)} after dedup")

df["PSA_Id"] = range(1, len(df) + 1)
df.to_csv("data/interim/grounded_generated_batch.csv", index=False)

print()
print("By domain:")
print(df["Domain"].value_counts())