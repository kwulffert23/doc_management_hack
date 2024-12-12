SELECT
  *
FROM
  VECTOR_SEARCH(
    index => "kyra_wulffert.poc_doc_management.db_doc_poc_doc_management_sequential",
    query => topic,
    num_results => 5
  )