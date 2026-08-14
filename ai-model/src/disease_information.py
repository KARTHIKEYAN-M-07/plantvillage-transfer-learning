"""
P1.5 - Disease Information
==========================

Provides structured informational content for the 38
PlantVillage classes used by the AI model.

IMPORTANT:
    This information is general educational guidance.
    It is not a substitute for professional agricultural
    diagnosis or pesticide recommendations.
"""


from __future__ import annotations

from typing import Any


# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================

DISEASE_INFORMATION: dict[str, dict[str, Any]] = {

    # ========================================================
    # APPLE
    # ========================================================

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple scab",
        "status": "diseased",

        "description": (
            "Apple scab is a fungal disease that commonly "
            "affects apple leaves and fruit."
        ),

        "symptoms": [
            "Olive-green to dark lesions on leaves",
            "Dark scabby lesions may develop on fruit",
            "Severe infections can cause premature leaf drop",
        ],

        "general_management": [
            "Remove and dispose of heavily affected plant material",
            "Maintain good air circulation around plants",
            "Avoid prolonged leaf wetness where possible",
            "Use locally recommended disease-management practices",
        ],

        "prevention": [
            "Maintain orchard sanitation",
            "Remove fallen infected leaves",
            "Prune appropriately to improve airflow",
            "Use disease-resistant varieties where available",
        ],

        "severity": "moderate_to_high",
    },


    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black rot",
        "status": "diseased",

        "description": (
            "Black rot is a fungal disease that can affect "
            "apple leaves, branches, and fruit."
        ),

        "symptoms": [
            "Circular brown or purple-brown leaf spots",
            "Expanding dark lesions",
            "Fruit may develop dark rotting areas",
        ],

        "general_management": [
            "Remove infected plant material",
            "Maintain orchard sanitation",
            "Prune affected dead wood",
            "Follow locally recommended disease-management practices",
        ],

        "prevention": [
            "Remove mummified fruit",
            "Remove dead or infected branches",
            "Maintain good canopy airflow",
            "Monitor plants regularly",
        ],

        "severity": "moderate_to_high",
    },


    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar apple rust",
        "status": "diseased",

        "description": (
            "Cedar apple rust is a fungal disease that affects "
            "apple foliage and fruit."
        ),

        "symptoms": [
            "Yellow-orange spots on leaves",
            "Orange or rust-colored lesions",
            "Premature leaf drop in severe cases",
        ],

        "general_management": [
            "Remove severely affected plant material",
            "Improve air circulation",
            "Monitor symptoms throughout the growing season",
            "Use locally recommended management practices",
        ],

        "prevention": [
            "Use resistant cultivars when available",
            "Maintain orchard sanitation",
            "Monitor nearby alternate hosts",
            "Inspect new growth regularly",
        ],

        "severity": "moderate",
    },


    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with a healthy "
            "apple plant leaf."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor the plant regularly for changes",
        ],

        "prevention": [
            "Maintain appropriate irrigation",
            "Provide adequate nutrition",
            "Maintain good airflow",
            "Regularly inspect leaves and fruit",
        ],

        "severity": "none",
    },


    # ========================================================
    # BLUEBERRY
    # ========================================================

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with a healthy "
            "blueberry plant."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor regularly for disease symptoms",
        ],

        "prevention": [
            "Maintain appropriate irrigation",
            "Maintain good airflow",
            "Remove damaged plant material",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # CHERRY
    # ========================================================

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery mildew",
        "status": "diseased",

        "description": (
            "Powdery mildew is a fungal disease characterized "
            "by powder-like growth on plant surfaces."
        ),

        "symptoms": [
            "White powdery growth",
            "Distorted young leaves",
            "Reduced plant vigor",
        ],

        "general_management": [
            "Remove severely affected plant material",
            "Improve air circulation",
            "Avoid excessive humidity around foliage",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Maintain good canopy airflow",
            "Avoid excessive nitrogen fertilization",
            "Monitor young growth regularly",
        ],

        "severity": "moderate",
    },


    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with a healthy "
            "cherry plant."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain adequate irrigation",
            "Maintain good airflow",
            "Remove damaged plant material",
        ],

        "severity": "none",
    },


    # ========================================================
    # CORN
    # ========================================================

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora leaf spot / Gray leaf spot",
        "status": "diseased",

        "description": (
            "Gray leaf spot is a fungal disease that produces "
            "elongated lesions on corn leaves."
        ),

        "symptoms": [
            "Long rectangular gray or tan lesions",
            "Lesions commonly follow leaf veins",
            "Reduced photosynthetic leaf area",
        ],

        "general_management": [
            "Monitor fields regularly",
            "Maintain appropriate crop residue management",
            "Use locally recommended disease-management practices",
        ],

        "prevention": [
            "Use resistant hybrids where available",
            "Practice crop rotation where appropriate",
            "Monitor disease development",
        ],

        "severity": "moderate_to_high",
    },


    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common rust",
        "status": "diseased",

        "description": (
            "Common rust is a fungal disease producing "
            "rust-colored pustules on corn leaves."
        ),

        "symptoms": [
            "Small reddish-brown pustules",
            "Pustules may occur on both leaf surfaces",
            "Severe infection can reduce photosynthetic capacity",
        ],

        "general_management": [
            "Monitor plants during favorable weather",
            "Use locally recommended disease-management practices",
            "Consider resistant hybrids where available",
        ],

        "prevention": [
            "Use resistant varieties",
            "Monitor fields regularly",
            "Maintain appropriate crop management",
        ],

        "severity": "moderate",
    },


    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "status": "diseased",

        "description": (
            "Northern Leaf Blight is a fungal disease that "
            "causes elongated lesions on corn leaves."
        ),

        "symptoms": [
            "Long gray-green or tan lesions",
            "Lesions can become large and cigar-shaped",
            "Reduced green leaf area",
        ],

        "general_management": [
            "Monitor fields regularly",
            "Use resistant hybrids where available",
            "Follow locally recommended disease-management practices",
        ],

        "prevention": [
            "Use resistant varieties",
            "Practice crop rotation where appropriate",
            "Manage crop residue appropriately",
        ],

        "severity": "moderate_to_high",
    },


    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy corn foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal crop management",
            "Monitor plants regularly",
        ],

        "prevention": [
            "Maintain adequate nutrition",
            "Manage irrigation appropriately",
            "Inspect foliage regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # GRAPE
    # ========================================================

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black rot",
        "status": "diseased",

        "description": (
            "Black rot is a fungal disease affecting grape "
            "leaves, shoots, and fruit."
        ),

        "symptoms": [
            "Brown circular leaf spots",
            "Dark lesions",
            "Fruit can develop dark rot",
        ],

        "general_management": [
            "Remove infected plant material",
            "Maintain vineyard sanitation",
            "Improve canopy airflow",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Remove infected or mummified fruit",
            "Maintain good canopy management",
            "Monitor regularly",
        ],

        "severity": "moderate_to_high",
    },


    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "status": "diseased",

        "description": (
            "Esca is a complex grapevine disease associated "
            "with fungal pathogens."
        ),

        "symptoms": [
            "Interveinal leaf discoloration",
            "Tiger-stripe patterns may occur",
            "Fruit symptoms may develop",
        ],

        "general_management": [
            "Remove severely affected plant material where appropriate",
            "Maintain vineyard sanitation",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Use healthy planting material",
            "Protect pruning wounds where recommended",
            "Monitor vines regularly",
        ],

        "severity": "high",
    },


    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf blight (Isariopsis Leaf Spot)",
        "status": "diseased",

        "description": (
            "A fungal leaf disease that produces spots and "
            "blight symptoms on grape foliage."
        ),

        "symptoms": [
            "Dark leaf spots",
            "Expanding lesions",
            "Leaf tissue deterioration",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Improve canopy airflow",
            "Maintain vineyard sanitation",
        ],

        "prevention": [
            "Monitor foliage regularly",
            "Maintain good canopy management",
            "Avoid prolonged leaf wetness where possible",
        ],

        "severity": "moderate",
    },


    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy grape foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal vineyard care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain canopy airflow",
            "Maintain adequate irrigation",
            "Inspect leaves regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # ORANGE
    # ========================================================

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus greening)",
        "status": "diseased",

        "description": (
            "Huanglongbing is a serious citrus disease that "
            "can cause leaf discoloration and decline."
        ),

        "symptoms": [
            "Blotchy or asymmetric leaf yellowing",
            "Leaf vein discoloration",
            "General decline in plant vigor",
        ],

        "general_management": [
            "Obtain professional confirmation of suspected cases",
            "Monitor trees and vectors",
            "Follow regional citrus disease-management guidance",
        ],

        "prevention": [
            "Use certified healthy planting material",
            "Monitor and manage insect vectors according to local guidance",
            "Remove infected trees where officially recommended",
        ],

        "severity": "high",
    },


    # ========================================================
    # PEACH
    # ========================================================

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial spot",
        "status": "diseased",

        "description": (
            "Bacterial spot is a bacterial disease that can "
            "affect peach leaves and fruit."
        ),

        "symptoms": [
            "Small dark leaf spots",
            "Lesions may become angular",
            "Fruit spots or surface lesions",
        ],

        "general_management": [
            "Remove severely affected material",
            "Maintain good orchard sanitation",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Use resistant cultivars where available",
            "Maintain good airflow",
            "Avoid unnecessary leaf wetness",
        ],

        "severity": "moderate",
    },


    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy peach foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal orchard care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain adequate nutrition",
            "Maintain airflow",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # PEPPER
    # ========================================================

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper",
        "disease": "Bacterial spot",
        "status": "diseased",

        "description": (
            "Bacterial spot can cause lesions on pepper leaves "
            "and fruit."
        ),

        "symptoms": [
            "Small dark leaf spots",
            "Brown or black lesions",
            "Fruit lesions may develop",
        ],

        "general_management": [
            "Remove severely affected material",
            "Avoid working with wet foliage",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Use clean planting material",
            "Avoid overhead irrigation where appropriate",
            "Maintain field sanitation",
        ],

        "severity": "moderate",
    },


    "Pepper,_bell___healthy": {
        "plant": "Pepper",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy pepper foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain adequate irrigation",
            "Maintain good airflow",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # POTATO
    # ========================================================

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early blight",
        "status": "diseased",

        "description": (
            "Early blight is a fungal disease commonly associated "
            "with characteristic leaf lesions."
        ),

        "symptoms": [
            "Brown circular lesions",
            "Concentric ring patterns may occur",
            "Older leaves are often affected first",
        ],

        "general_management": [
            "Remove heavily affected plant material",
            "Maintain appropriate field sanitation",
            "Follow locally recommended disease-management practices",
        ],

        "prevention": [
            "Practice crop rotation where appropriate",
            "Maintain balanced plant nutrition",
            "Avoid prolonged leaf wetness",
        ],

        "severity": "moderate_to_high",
    },


    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late blight",
        "status": "diseased",

        "description": (
            "Late blight is a serious disease that can rapidly "
            "damage potato foliage and tubers."
        ),

        "symptoms": [
            "Water-soaked dark lesions",
            "Rapidly expanding leaf damage",
            "Dark lesions on stems or tubers may occur",
        ],

        "general_management": [
            "Seek prompt local agricultural guidance",
            "Remove severely affected material where appropriate",
            "Follow regional late-blight management recommendations",
        ],

        "prevention": [
            "Use resistant varieties where available",
            "Monitor weather and disease conditions",
            "Avoid prolonged leaf wetness",
        ],

        "severity": "high",
    },


    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy potato foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal crop care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain balanced nutrition",
            "Manage irrigation appropriately",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # RASPBERRY
    # ========================================================

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy raspberry foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain good airflow",
            "Maintain adequate irrigation",
            "Remove damaged plant material",
        ],

        "severity": "none",
    },


    # ========================================================
    # SOYBEAN
    # ========================================================

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy soybean foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal crop care",
            "Monitor plants regularly",
        ],

        "prevention": [
            "Maintain balanced nutrition",
            "Monitor for disease symptoms",
            "Maintain appropriate irrigation",
        ],

        "severity": "none",
    },


    # ========================================================
    # SQUASH
    # ========================================================

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery mildew",
        "status": "diseased",

        "description": (
            "Powdery mildew is a fungal disease producing "
            "white powder-like growth on leaves."
        ),

        "symptoms": [
            "White powdery patches",
            "Yellowing or declining leaves",
            "Reduced photosynthetic capacity",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Improve airflow around plants",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Avoid excessive humidity",
            "Maintain plant spacing",
            "Monitor leaves regularly",
        ],

        "severity": "moderate",
    },


    # ========================================================
    # STRAWBERRY
    # ========================================================

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf scorch",
        "status": "diseased",

        "description": (
            "Leaf scorch is associated with dark lesions that "
            "can cause portions of strawberry leaves to die."
        ),

        "symptoms": [
            "Dark purple to brown leaf spots",
            "Spots can enlarge and cause tissue death",
            "Affected leaves may appear scorched",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Maintain field sanitation",
            "Improve airflow",
        ],

        "prevention": [
            "Monitor plants regularly",
            "Avoid prolonged leaf wetness",
            "Use healthy planting material",
        ],

        "severity": "moderate",
    },


    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy strawberry foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor leaves regularly",
        ],

        "prevention": [
            "Maintain adequate irrigation",
            "Maintain airflow",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },


    # ========================================================
    # TOMATO
    # ========================================================

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial spot",
        "status": "diseased",

        "description": (
            "Bacterial spot is a bacterial disease that can "
            "produce dark lesions on tomato leaves and fruit."
        ),

        "symptoms": [
            "Small dark spots on leaves",
            "Lesions may become irregular",
            "Fruit lesions may occur",
        ],

        "general_management": [
            "Remove severely affected material",
            "Maintain field sanitation",
            "Avoid handling plants when foliage is wet",
        ],

        "prevention": [
            "Use clean planting material",
            "Avoid unnecessary overhead irrigation",
            "Maintain good airflow",
        ],

        "severity": "moderate",
    },


    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early blight",
        "status": "diseased",

        "description": (
            "Early blight is a fungal disease that commonly "
            "causes dark lesions on tomato foliage."
        ),

        "symptoms": [
            "Brown circular leaf lesions",
            "Concentric ring patterns",
            "Yellowing around lesions",
            "Older leaves may be affected first",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Maintain field sanitation",
            "Avoid prolonged leaf wetness",
            "Follow locally recommended management practices",
        ],

        "prevention": [
            "Practice crop rotation where appropriate",
            "Maintain good airflow",
            "Use healthy planting material",
            "Monitor plants regularly",
        ],

        "severity": "moderate_to_high",
    },


    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late blight",
        "status": "diseased",

        "description": (
            "Late blight is a serious disease capable of "
            "rapidly damaging tomato foliage and fruit."
        ),

        "symptoms": [
            "Dark water-soaked lesions",
            "Rapidly expanding leaf damage",
            "Dark lesions on fruit may occur",
        ],

        "general_management": [
            "Seek prompt local agricultural guidance",
            "Remove affected material where appropriate",
            "Follow regional disease-management recommendations",
        ],

        "prevention": [
            "Monitor weather and disease conditions",
            "Use resistant varieties where available",
            "Avoid prolonged leaf wetness",
        ],

        "severity": "high",
    },


    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "status": "diseased",

        "description": (
            "Tomato leaf mold is a fungal disease favored by "
            "high humidity."
        ),

        "symptoms": [
            "Yellow patches on upper leaf surfaces",
            "Olive or gray fungal growth underneath leaves",
            "Leaf decline in severe infections",
        ],

        "general_management": [
            "Improve ventilation",
            "Reduce excessive humidity",
            "Remove severely affected leaves",
        ],

        "prevention": [
            "Maintain good airflow",
            "Avoid prolonged leaf wetness",
            "Monitor greenhouse humidity where applicable",
        ],

        "severity": "moderate",
    },


    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria leaf spot",
        "status": "diseased",

        "description": (
            "Septoria leaf spot is a fungal disease that "
            "primarily affects tomato leaves."
        ),

        "symptoms": [
            "Small circular leaf spots",
            "Dark margins around lesions",
            "Small dark structures may appear in lesion centers",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Maintain sanitation",
            "Avoid prolonged leaf wetness",
        ],

        "prevention": [
            "Practice crop rotation where appropriate",
            "Maintain good airflow",
            "Remove infected plant debris",
        ],

        "severity": "moderate",
    },


    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider mites (Two-spotted spider mite)",
        "status": "diseased",

        "description": (
            "Two-spotted spider mites are small pests that feed "
            "on plant cells and can cause leaf discoloration."
        ),

        "symptoms": [
            "Fine yellow or pale stippling",
            "Bronzing or yellowing leaves",
            "Fine webbing may occur with heavier infestations",
        ],

        "general_management": [
            "Inspect leaf undersides",
            "Use locally recommended integrated pest-management practices",
            "Avoid unnecessary broad-spectrum pesticide use",
        ],

        "prevention": [
            "Monitor plants regularly",
            "Maintain appropriate plant moisture",
            "Encourage beneficial predators where appropriate",
        ],

        "severity": "moderate",
    },


    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "status": "diseased",

        "description": (
            "Target spot is a fungal disease that causes circular "
            "lesions on tomato foliage."
        ),

        "symptoms": [
            "Circular brown lesions",
            "Concentric target-like rings",
            "Leaf yellowing and decline",
        ],

        "general_management": [
            "Remove severely affected leaves",
            "Maintain good field sanitation",
            "Improve airflow",
        ],

        "prevention": [
            "Avoid prolonged leaf wetness",
            "Maintain plant spacing",
            "Monitor foliage regularly",
        ],

        "severity": "moderate",
    },


    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato mosaic virus",
        "status": "diseased",

        "description": (
            "Tomato mosaic virus is a viral disease that can "
            "cause mosaic patterns and growth abnormalities."
        ),

        "symptoms": [
            "Mottled light and dark green leaf patterns",
            "Leaf distortion",
            "Reduced plant growth",
        ],

        "general_management": [
            "Remove severely affected plants where appropriate",
            "Avoid spreading plant sap between plants",
            "Follow local plant-health guidance",
        ],

        "prevention": [
            "Use clean planting material",
            "Disinfect tools appropriately",
            "Control sources of mechanical transmission",
        ],

        "severity": "high",
    },


    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "status": "diseased",

        "description": (
            "Tomato Yellow Leaf Curl Virus can cause severe "
            "leaf curling, yellowing, and plant growth reduction."
        ),

        "symptoms": [
            "Upward curling leaves",
            "Leaf yellowing",
            "Reduced plant growth",
            "Stunting in severe cases",
        ],

        "general_management": [
            "Monitor and manage insect vectors according to local guidance",
            "Remove severely affected plants where recommended",
            "Follow local plant-health recommendations",
        ],

        "prevention": [
            "Use healthy planting material",
            "Monitor vector populations",
            "Use suitable resistant varieties where available",
        ],

        "severity": "high",
    },


    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "status": "healthy",

        "description": (
            "The image appears consistent with healthy tomato foliage."
        ),

        "symptoms": [],

        "general_management": [
            "Continue normal plant care",
            "Monitor foliage regularly",
        ],

        "prevention": [
            "Maintain adequate irrigation",
            "Maintain balanced nutrition",
            "Maintain good airflow",
            "Inspect plants regularly",
        ],

        "severity": "none",
    },
}


# ============================================================
# EXPECTED PLANTVILLAGE CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]


# ============================================================
# VALIDATE DATABASE
# ============================================================

def validate_database() -> None:
    """
    Verify that all 38 PlantVillage classes have
    disease information.
    """

    missing = [
        class_name
        for class_name in EXPECTED_CLASSES
        if class_name not in DISEASE_INFORMATION
    ]

    extra = [
        class_name
        for class_name in DISEASE_INFORMATION
        if class_name not in EXPECTED_CLASSES
    ]

    if missing:

        raise RuntimeError(
            "Missing disease information for:\n"
            + "\n".join(missing)
        )

    if extra:

        raise RuntimeError(
            "Unexpected disease information entries:\n"
            + "\n".join(extra)
        )

    if len(DISEASE_INFORMATION) != 38:

        raise RuntimeError(
            f"Expected 38 disease entries, "
            f"found {len(DISEASE_INFORMATION)}."
        )


# ============================================================
# GET INFORMATION
# ============================================================

def get_disease_information(
    class_name: str,
) -> dict[str, Any]:
    """
    Return information for a PlantVillage class.

    Raises:
        ValueError if the class is unknown.
    """

    validate_database()

    if class_name not in DISEASE_INFORMATION:

        raise ValueError(
            f"Unknown PlantVillage class: "
            f"{class_name}"
        )

    information = DISEASE_INFORMATION[
        class_name
    ]

    return {
        "disease_information": {
            "class_name": class_name,
            **information,
        }
    }


# ============================================================
# HEALTHY CHECK
# ============================================================

def is_healthy_class(
    class_name: str,
) -> bool:

    information = (
        get_disease_information(
            class_name
        )
    )

    return (
        information[
            "disease_information"
        ]["status"]
        == "healthy"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src\\disease_information.py "
            "<class_name>"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python src\\disease_information.py "
            "\"Apple___Apple_scab\""
        )

        raise SystemExit(1)

    class_name = sys.argv[1]

    result = get_disease_information(
        class_name
    )

    information = (
        result[
            "disease_information"
        ]
    )

    print("=" * 60)

    print(
        "P1.5 DISEASE INFORMATION"
    )

    print("=" * 60)

    print(
        f"Class: "
        f"{information['class_name']}"
    )

    print(
        f"Plant: "
        f"{information['plant']}"
    )

    print(
        f"Disease: "
        f"{information['disease']}"
    )

    print(
        f"Status: "
        f"{information['status']}"
    )

    print(
        f"Severity: "
        f"{information['severity']}"
    )

    print()

    print(
        "Description:"
    )

    print(
        information["description"]
    )

    print()

    print(
        "Symptoms:"
    )

    if information["symptoms"]:

        for item in information["symptoms"]:

            print(
                f"  - {item}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "General Management:"
    )

    for item in information[
        "general_management"
    ]:

        print(
            f"  - {item}"
        )

    print()

    print(
        "Prevention:"
    )

    for item in information[
        "prevention"
    ]:

        print(
            f"  - {item}"
        )

    print("=" * 60)