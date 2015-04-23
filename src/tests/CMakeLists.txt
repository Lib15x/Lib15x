macro (add_test_by_fail_regex name regex arguments)
  add_executable_by_name(${name})
  add_test (${name}Test ${name} ${arguments})
  set_tests_properties ("${name}Test"
    PROPERTIES FAIL_REGULAR_EXPRESSION ${regex})
endmacro (add_test_by_fail_regex)

add_test_by_fail_regex(LibSVM_unit "test failed" "")
add_test_by_fail_regex(SupportVectorClassifier_unit "test failed" "")
add_test_by_fail_regex(Scaler_unit "test failed" "")
