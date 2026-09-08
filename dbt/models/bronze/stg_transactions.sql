-- =======================================================================
-- Couche : BRONZE (Staging)
-- Objectif : Nettoyer la donnée brute (typage, renommage) sans changer les valeurs.
-- =======================================================================

WITH raw_data AS (
    -- On va lire directement la table raw_transactions dans la base par défaut
    SELECT * FROM default.raw_transactions
)

SELECT
    -- Renommage en snake_case (standard SQL) pour plus de lisibilité
    step,
    type AS transaction_type,
    amount AS transaction_amount,
    
    -- Informations sur l'émetteur
    nameOrig AS origin_account_id,
    oldbalanceOrg AS origin_old_balance,
    newbalanceOrig AS origin_new_balance,
    
    -- Informations sur le destinataire
    nameDest AS destination_account_id,
    oldbalanceDest AS destination_old_balance,
    newbalanceDest AS destination_new_balance,
    
    -- Conversion explicite des indicateurs de fraude
    CAST(isFraud AS Boolean) AS is_fraud,
    CAST(isFlaggedFraud AS Boolean) AS is_flagged_fraud

FROM raw_data
