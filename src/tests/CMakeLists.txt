macro (add_test_by_fail_regex name regex arguments)
  add_executable_by_name(${name})
  add_test (${name}Test ${name} ${arguments})
  set_tests_properties ("${name}Test"
    PROPERTIES FAIL_REGULAR_EXPRESSION ${regex})
endmacro (add_test_by_fail_regex)


# element tests
add_test_by_fail_regex(Main "test failed" "")
