(
  SELECT 
    definition,
    certainty
  FROM kyra_wulffert.poc_doc_management.normalised_glossary
  WHERE acronym ILIKE TRIM(input_acronym)
    AND certainty >= 80
  ORDER BY certainty DESC
)