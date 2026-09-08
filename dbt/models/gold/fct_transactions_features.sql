-- =======================================================================
-- Couche : GOLD (Marts)
-- Objectif : Table finale, ultra-optimisée, contenant uniquement 
-- les colonnes pertinentes pour entraîner notre algorithme XGBoost (Jour 3).
-- =======================================================================

WITH silver AS (
    SELECT * FROM {{ ref('int_fraud_features') }}
)

SELECT
    -- On garde le temps (utile pour des validations croisées temporelles)
    step,
    
    -- Variables Catégorielles (Type de transaction)
    transaction_type,
    
    -- Variables Numériques (Montants et soldes)
    transaction_amount,
    origin_old_balance,
    origin_new_balance,
    destination_old_balance,
    destination_new_balance,
    
    -- Nos belles "Features" (Silver) construites intelligemment
    is_origin_emptied,
    origin_balance_error,
    destination_balance_error,
    amount_to_oldbalance_ratio,
    is_merchant_destination,
    
    -- La variable Cible (Target) que le ML doit apprendre à prédire
    is_fraud

    -- Remarque : On a EXCLU intentionnellement `origin_account_id` et `destination_account_id`
    -- Pourquoi ? Car ce sont des identifiants uniques (des chaînes de caractères arbitraires). 
    -- L'algorithme ne peut rien en apprendre de généralisable.
    
FROM silver
