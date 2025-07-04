from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

FIELD_VALUE_SCHEMA = {
    "event_type": [
        'VIOLENT CRIME', 'THEFT & BURGLARY', 'PUBLIC DISTURBANCE', 'FIRE & HAZARDS',
        'RESCUE OPERATIONS', 'MEDICAL EMERGENCIES', 'TRAFFIC INCIDENTS',
        'PUBLIC NUISANCE', 'SOCIAL ISSUES', 'MISSING PERSONS',
        'NATURAL INCIDENTS', 'OTHERS'
    ],
    "event_sub_type": {
        'VIOLENT CRIME': [
            'ASSAULT', 'KIDNAPPING', 'BOMB BLAST', 'CHILD ABUSE', 'CRIME AGAINST WOMEN',
            'ROBBERY', 'DEAD BODY FOUND', 'SUICIDE ATTEMPT', 'MURDER', 'THREAT',
            'DOMESTIC VIOLENCE', 'VERBAL ABUSE'
        ],
        'THEFT & BURGLARY': [
            'THEFT', 'ATTEMPT OF THEFT', 'HOUSE BREAKING ATTEMPTS', 'VEHICLE THEFT'
        ],
        'PUBLIC DISTURBANCE': [
            'SCUFFLE AMONG STUDENTS', 'DRUNKEN ATROCITIES', 'VERBAL ABUSE', 'GAMBLING',
            'NUDITY IN PUBLIC', 'STRIKE', 'GENERAL NUISANCE'
        ],
        'FIRE & HAZARDS': [
            'FIRE', 'ELECTRICAL FIRE', 'BUILDING FIRE', 'LANDSCAPE FIRE', 'VEHICLE FIRE',
            'GAS LEAKAGE', 'HAZARDOUS CONDITION INCIDENTS'
        ],
        'RESCUE OPERATIONS': [
            'WATER RESCUE', 'WELL RESCUE', 'SEARCH AND RESCUE', 'ROAD CRASH RESCUE'
        ],
        'MEDICAL EMERGENCIES': [
            'BREATHING DIFFICULTIES', 'PERSON COLLAPSED', 'AMBULANCE SERVICE',
            'INTER HOSPITAL TRANSFER', 'BLEEDING', 'HEART ATTACK', 'FIRE INJURY'
        ],
        'TRAFFIC INCIDENTS': [
            'HIT & RUN INCIDENTS', 'RASH DRIVING', 'TRAFFIC BLOCK', 'OBSTRUCTIVE PARKING',
            'VEHICLE BREAK DOWN', 'ACCIDENT', 'RUN OVER INCIDENTS'
        ],
        'PUBLIC NUISANCE': [
            'ILLEGAL MINING', 'ILLEGAL CONSTRUCTIONS', 'GENERAL NUISANCE',
            'WASTAGE DUMPING ISSUE', 'AIR POLLUTION', 'NOISE POLLUTION',
            'TRESPASSING TO PROPERTY'
        ],
        'SOCIAL ISSUES': [
            'FAMILY ISSUES', 'LABOUR/WAGES ISSUES', 'DISPUTES BETWEEN NEIGHBOURS',
            'LAND BOUNDARY ISSUES', 'ISSUES RELATED TO SENIOR CITIZENS',
            'MIGRANT LABOURERS ISSUES'
        ],
        'MISSING PERSONS': [
            'MISSING', 'SUSPICIOUSLY FOUND PERSONS OR VEHICLES', 'CHILD LINES'
        ],
        'NATURAL INCIDENTS': [
            'RAINY SEASON INCIDENTS', 'NIZHAL PANIC CALL', 'FLOOD', 'DISASTER',
            'EARTHQUAKE', 'LANDSLIDE', 'STRUCTURE COLLAPSE'
        ],
        'OTHERS': [
            'OTHERS', 'SALE OF CONTRABANDS', 'ASSISTANCE FOR HOSPITALIZATION OF CHALLENGED PERSONS',
            'CYBER CRIME', 'RAILWAY', 'ABANDONED VEHICLES'
        ]
    },
    "state_of_victim": [
        'Distressed', 'Stable', 'Injured', 'Critical', 'Unconscious', 'Deceased', 'Drunken', 'not specified'
    ],
    "victim_gender": [
        'male', 'female', 'not specified'
    ]
}

# --- MOVE THESE DEFINITIONS UP ---
# Ensure ALL_EVENT_TYPES is defined before ProcessedOutput uses it
ALL_EVENT_TYPES = FIELD_VALUE_SCHEMA["event_type"]

# Ensure ALL_EVENT_SUB_TYPES is defined before ProcessedOutput uses it
ALL_EVENT_SUB_TYPES = sorted(list(set(
    item for sublist in FIELD_VALUE_SCHEMA["event_sub_type"].values() for item in sublist
)))
# --- END MOVE ---


# List of fields that are expected to be present in the output
ALL_CLASSIFICATION_FIELDS = [
    "event_type", "event_sub_type", "state_of_victim", "victim_gender",
    "specified_matter", "date_reference", "frequency", "repeat_incident",
    "identification", "injury_type", "victim_age", "victim_relation",
    "incident_location", "area", "suspect_description", "object_involved",
    "used_weapons", "offender_relation", "mode_of_threat", "need_ambulance",
    "children_involved"
]

def derive_event_type(sub_type: str) -> str:
    """Derive event_type from event_sub_type, handling 'OTHERS' specially."""
    sub_type_upper = sub_type.upper()
    
    # If the sub_type is "OTHERS" (meaning it was originally "OTHERS: <label>" or just "OTHERS"),
    # then its event_type is always "OTHERS".
    if sub_type_upper == 'OTHERS':
        return 'OTHERS'
        
    for event_type_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
        if sub_type_upper in [st.upper() for st in sub_types_list]:
            return event_type_key
    
    # Fallback for any sub_type not found (should ideally not happen with good LLM output)
    return 'OTHERS'


class ProcessedOutput(BaseModel):
    """Schema for processed text output, including processing metadata."""
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float
    file_name: str
    file_text: str
    event_type: Literal[tuple(ALL_EVENT_TYPES)]
    event_sub_type: Literal[tuple(ALL_EVENT_SUB_TYPES)] # This remains strict
    state_of_victim: Literal[tuple(FIELD_VALUE_SCHEMA["state_of_victim"])]
    victim_gender: Literal[tuple(FIELD_VALUE_SCHEMA["victim_gender"])]
    specified_matter: Optional[str] = None
    date_reference: Optional[str] = None
    frequency: Optional[str] = None
    repeat_incident: Optional[Literal["yes", "no", "not_specified", "not applicable", "not specified"]] = "not specified"
    identification: Optional[str] = None
    injury_type: Optional[str] = None
    victim_age: Optional[str] = None
    victim_relation: Optional[str] = None
    incident_location: Optional[str] = None
    area: Optional[str] = None
    suspect_description: Optional[str] = None
    object_involved: Optional[str] = None
    date_of_birth: Optional[str] = None
    used_weapons: Optional[str] = None
    offender_relation: Optional[str] = None
    mode_of_threat: Optional[str] = None
    need_ambulance: Optional[Literal["yes", "no", "not_specified", "not applicable", "not specified"]] = "not specified"
    children_involved: Optional[Literal["yes", "no", "not_specified", "not applicable", "not specified"]] = "not specified"

    # NEW FIELD: This will store the specific detail when event_sub_type is "OTHERS"
    generated_event_sub_type_detail: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-03-20T10:30:00",
                "processing_time": 1.5,
                "file_name": "emergency_call_001.txt",
                "file_text": "Emergency call reporting assault in building",
                "event_type": "VIOLENT CRIME",
                "event_sub_type": "ASSAULT",
                "state_of_victim": "Injured",
                "victim_gender": "male",
                "incident_location": "123 Main St",
                "area": "Downtown",
                "specified_matter": "Physical assault with injuries",
                "date_reference": None,
                "date_of_birth": None,
                "generated_event_sub_type_detail": None # Example for the new field
            }
        }

    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event_type"""
        # Ensure 'OTHERS' is also covered by upper casing for validation consistency
        if v.upper() not in [et.upper() for et in FIELD_VALUE_SCHEMA["event_type"]]:
            raise ValueError(f"Invalid event_type: {v}")
        return v # Return original casing, or v.upper() if your system strictly uses upper

    @field_validator('event_sub_type')
    @classmethod
    def validate_event_sub_type(cls, v: str) -> str:
        """Validate event_sub_type"""
        # This validator will only see 'OTHERS' or a valid ALL_EVENT_SUB_TYPES value
        if v.upper() not in [st.upper() for st in ALL_EVENT_SUB_TYPES]:
            raise ValueError(f"Invalid event_sub_type: {v}")
        return v # Return original casing, or v.upper()

    @field_validator('state_of_victim')
    @classmethod
    def validate_state_of_victim(cls, v: str) -> str:
        """Validate state_of_victim"""
        if v.upper() not in [sv.upper() for sv in FIELD_VALUE_SCHEMA["state_of_victim"]]:
            raise ValueError(f"Invalid state_of_victim: {v}")
        return v # Return original casing, or v.upper()

    @field_validator('victim_gender')
    @classmethod
    def validate_victim_gender(cls, v: str) -> str:
        """Validate victim_gender"""
        if v.lower() not in [vg.lower() for vg in FIELD_VALUE_SCHEMA["victim_gender"]]:
            raise ValueError(f"Invalid victim_gender: {v}")
        return v # Return original casing, or v.lower()

# --- GroundTruthOutput Schema (No changes needed, but included for completeness) ---
class GroundTruthOutput(BaseModel):
    """
    Represents the ground truth data, containing only the extracted information
    that would be provided as a gold standard. No processing metadata here.
    """
    # Required fields based on the structure of your ground truth data
    event_type: str
    event_sub_type: str
    state_of_victim: str = Field("not specified", description="State of the victim")
    victim_gender: str = Field("not specified", description="Gender of the victim")
    
    # Optional text fields (matching ProcessedOutput's optional fields)
    specified_matter: Optional[str] = None
    date_reference: Optional[str] = None
    frequency: Optional[str] = None
    repeat_incident: Optional[str] = None
    identification: Optional[str] = None
    injury_type: Optional[str] = None
    victim_age: Optional[str] = None
    victim_relation: Optional[str] = None
    incident_location: Optional[str] = None
    area: Optional[str] = None
    suspect_description: Optional[str] = None
    object_involved: Optional[str] = None
    date_of_birth: Optional[str] = None
    used_weapons: Optional[str] = None
    offender_relation: Optional[str] = None
    mode_of_threat: Optional[str] = None
    need_ambulance: Optional[str] = None
    children_involved: Optional[str] = None
    generated_event_sub_type_detail: Optional[str] = None 