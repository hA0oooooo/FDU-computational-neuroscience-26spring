from src.models_brainiac import (  # noqa: F401
    BrainIACEncoder,
    BrainIACLabelModel,
    BrainIACTaskModel,
    checkpoint_missing_message,
    extract_embeddings,
    load_brainiac_backbone_class,
    load_embedding_ids,
    save_embeddings,
)


BrainIACAgeSexModel = BrainIACTaskModel
