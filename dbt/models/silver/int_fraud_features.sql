-- =======================================================================
-- Couche : SILVER (Intermediate)
-- Objectif : Calculer de nouvelles "features" (indicateurs) pour le Machine Learning.
-- C'est ici que l'on commence à ajouter de la logique métier (Feature Engineering).
-- =======================================================================

WITH staging AS (
    -- On se base sur la couche Bronze qui est maintenant propre
    SELECT * FROM {{ ref('stg_transactions') }}
)

SELECT
    *,
    
    -- FEATURE 1 : Est-ce que le compte d'origine a été vidé exactement à 0 ?
    -- (Indicateur très fort de fraude type "Account Takeover" / piratage)
    CASE 
        WHEN origin_old_balance > 0 AND origin_new_balance = 0 THEN 1 
        ELSE 0 
    END AS is_origin_emptied,

    -- FEATURE 2 : Incohérence mathématique sur le compte source
    -- Si Ancien Solde - Montant != Nouveau Solde, c'est étrange (erreur ou fraude complexe)
    -- On arrondit pour éviter les erreurs de virgule flottante
    ROUND(origin_old_balance - transaction_amount - origin_new_balance, 2) AS origin_balance_error,

    -- FEATURE 3 : Incohérence mathématique sur le compte destinataire
    -- Si Ancien Solde + Montant != Nouveau Solde
    ROUND(destination_old_balance + transaction_amount - destination_new_balance, 2) AS destination_balance_error,

    -- FEATURE 4 : Ratio du montant par rapport au solde d'origine
    -- Si la personne transfère 100% (1.0) de son compte, c'est très différent de si elle transfère 1%
    CASE 
        WHEN origin_old_balance = 0 THEN 0 -- Évite la division par zéro
        ELSE ROUND(transaction_amount / origin_old_balance, 4)
    END AS amount_to_oldbalance_ratio,

    -- FEATURE 5 : Est-ce que le destinataire est un marchand ?
    -- L'IA comprendra ainsi pourquoi son solde est toujours à 0
    CASE 
        WHEN destination_account_id LIKE 'M%' THEN 1 
        ELSE 0 
    END AS is_merchant_destination

FROM staging
