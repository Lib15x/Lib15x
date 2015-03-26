macro (add_test_by_fail_regex name regex arguments)
  add_executable_by_name(${name})
  add_test (${name}Test ${name} ${arguments})
  set_tests_properties ("${name}Test"
    PROPERTIES FAIL_REGULAR_EXPRESSION ${regex})
endmacro (add_test_by_fail_regex)

add_test_by_fail_regex(DenseStaticMatrix_unit "test failed" "")
add_test_by_fail_regex(DenseDynamicMatrix_unit "test failed" "")
