import os

# Constants related to figures
FIGURE_WIDTH = 30
FIGURE_HEIGHT = 24
TITLE_FONTSIZE = 24
TITLE_PAD = 20
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
LABEL_PAD = 15
XTICK_FONTSIZE = 10
YTICK_FONTSIZE = 14
BAR_WIDTH = 0.6
TIGHT_LAYOUT_PAD = 8.0
TEXT_FONTSIZE = 5
DPI = 300
YLIM_MULTIPLIER = 1.15

# Constants related to correlation matrix
CORRELATION_FIGURE_WIDTH = 40
CORRELATION_FIGURE_HEIGHT = 30
CORRELATION_TITLE_FONTSIZE = 24
CORRELATION_XTICK_FONTSIZE = 10
CORRELATION_YTICK_FONTSIZE = 10
CORRELATION_TIGHT_LAYOUT_PAD = 3.0
CORRELATION_ANNOT_FONTSIZE = 8

# Define the path to the data file
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2017-2020', 'processed', 'FilteredCombinedData.csv')

TARGET_COLUMN = "BPQ020"

# Column definitions
COLUMN_DEFINITIONS = {
    "SEQN": "Respondent sequence number",
    "ALQ111": "Ever had a drink of any kind of alcohol",
    "ALQ121": "Past 12 mo how often drink alcoholic bev",
    "ALQ130": "Avg # alcoholic drinks/day - past 12 mos",
    "ALQ142": "# days have 4 or 5 drinks/past 12 mos",
    "ALQ270": "# times 4-5 drinks in 2hrs/past 12 mos",
    "ALQ280": "# times 8+ drinks in 1 day/past 12 mos",
    "ALQ290": "# times 12+ drinks in 1 day/past 12 mos",
    "ALQ151": "Ever have 4/5 or more drinks every day?",
    "ALQ170CK": "CHECK ITEM IF ALQ121 = 0, GO TO THE END OF SECTION. OTHERWISE, CONTINUE.",
    "ALQ170": "Past 30 days # times 4-5 drinks on an oc",
    "BPQ020": "Ever told you had high blood pressure",
    "BPQ030": "Told had high blood pressure - 2+ times",
    "BPD035": "Age told had hypertension",
    "BPQ040A": "Taking prescription for hypertension",
    "BPQ050A": "Now taking prescribed medicine for HBP",
    "BPQ080": "Doctor told you - high cholesterol level",
    "BPQ060": "Ever had blood cholesterol checked",
    "BPQ070": "When blood cholesterol last checked",
    "BPQ090D": "Told to take prescriptn for cholesterol",
    "BPQ100D": "Now taking prescribed medicine",
    "DBQ010": "Ever breastfed or fed breastmilk",
    "DBD030": "Age stopped breastfeeding(days)",
    "DBD041": "Age first fed formula(days)",
    "DBD050": "Age stopped receiving formula(days)",
    "DBD055": "Age started other food/beverage",
    "DBD061": "Age first fed milk(days)",
    "DBQ073A": "Type of milk first fed - whole milk",
    "DBQ073B": "Type of milk first fed - 2% milk",
    "DBQ073C": "Type of milk first fed - 1% milk",
    "DBQ073D": "Type of milk first fed - fat free milk",
    "DBQ073E": "Type of milk first fed - soy milk",
    "DBQ073U": "Type of milk first fed - other",
    "DBD085": "CHECK ITEM DBD085: IF SP AGE < 1 GO TO END OF SECTION, ELSE SP AGE 1-15 GO TO DBQ197, OTHERWISE CONTINUE.",
    "DBQ700": "How healthy is the diet",
    "DBQ197": "Past 30 day milk product consumption",
    "DBQ223A": "You drink whole or regular milk",
    "DBQ223B": "You drink 2% fat milk",
    "DBQ223C": "You drink 1% fat milk",
    "DBQ223D": "You drink fat free/skim milk",
    "DBQ223E": "You drink soy milk",
    "DBQ223U": "You drink another type of milk",
    "DBD225": "CHECK ITEM DBD225: IF SP AGE 1-19 GO TO DBD355, OTHERWISE CONTINUE.",
    "DBQ229": "Regular milk use 5 times per week",
    "DBQ235A": "How often drank milk age 5-12",
    "DBQ235B": "How often drank milk age 13-17",
    "DBQ235C": "How often drank milk age 18-35",
    "DBD265a": "CHECK ITEM DBD265A: IF SP AGE 20-59, GO TO DBD895, OTHERWISE CONTINUE. ",
    "DBQ301": "Community/Government meals delivered",
    "DBQ330": "Eat meals at Community/Senior center",
    "DBD355": "CHECK ITEM DBD355: IF SP AGE >=60 or <4, GO TO DBD895, OTHERWISE, CONTINUE.",
    "DBQ360": "Attend kindergarten thru high school",
    "DBQ370": "School serves school lunches",
    "DBD381": "# of times/week get school lunch",
    "DBQ390": "School lunch free, reduced or full price",
    "DBQ400": "School serve complete breakfast each day",
    "DBD411": "# of times/week get school breakfast",
    "DBQ421": "School breakfast free/reduced/full price",
    "DBQ422": "CHECK ITEM DBQ422: IF DBQ390 = CODE 1 OR CODE 2, OR DBQ421 = CODE 1 OR CODE 2, CONTINUE; OTHERWISE, GO TO DBD895.",
    "DBQ424": "Summer program meal free/reduced price",
    "DBD895": "# of meals not home prepared",
    "DBD900": "# of meals from fast food or pizza place",
    "DBD905": "# of ready-to-eat foods in past 30 days",
    "DBD910": "# of frozen meals/pizza in past 30 days",
    "DBQ715a": "CHECK ITEM DBQ715a: IF SP AGE < 16, GO TO END OF SECTION. OTHERWISE, CONTINUE.",
    "CBQ596": "Heard of My Plate",
    "CBQ606": "Looked up My Plate on internet",
    "CBQ611": "Tried My Plate plan",
    "DBQ930": "Main meal planner/preparer",
    "DBQ935": "Shared meal planning/preparing duty",
    "DBQ940": "Main food shopper",
    "DBQ945": "Shared food shopping duty",
    "PAQ605": "Vigorous work activity",
    "PAQ610": "Number of days vigorous work",
    "PAD615": "Minutes vigorous-intensity work",
    "PAQ620": "Moderate work activity",
    "PAQ625": "Number of days moderate work",
    "PAD630": "Minutes moderate-intensity work",
    "PAQ635": "Walk or bicycle",
    "PAQ640": "Number of days walk or bicycle",
    "PAD645": "Minutes walk/bicycle for transportation",
    "PAQ650": "Vigorous recreational activities",
    "PAQ655": "Days vigorous recreational activities",
    "PAD660": "Minutes vigorous recreational activities",
    "PAQ665": "Moderate recreational activities",
    "PAQ670": "Days moderate recreational activities",
    "PAD675": "Minutes moderate recreational activities",
    "PAD680": "Minutes sedentary activity",
    "SMQ020": "Smoked at least 100 cigarettes in life",
    "SMD030": "Age started smoking cigarettes regularly",
    "SMQ040": "Do you now smoke cigarettes?",
    "SMQ050Q": "How long since quit smoking cigarettes",
    "SMQ050U": "Unit of measure (day/week/month/year)",
    "SMD057": "# cigarettes smoked per day when quit",
    "SMQ078": "How soon after waking do you smoke",
    "SMD641": "# days smoked cigs during past 30 days",
    "SMD650": "Avg # cigarettes/day during past 30 days",
    "SMD100FL": "Cigarette Filter type",
    "SMD100MN": "Cigarette Menthol indicator",
    "SMQ670": "Tried to quit smoking",
    "SMQ621": "Cigarettes smoked in entire life",
    "SMD630": "Age first smoked whole cigarette",
    "SMAQUEX2": "Questionnaire Mode Flag"
}