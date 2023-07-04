import test_model

def get_next_action_func(state, difficulty):
    state_list = list(state)
    result = test_model.test_main(state_list, difficulty)
    print(result)
    return result

payload= {
  "list": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      2,
      0
    ],
    [
      0,
      1,
      0,
      0,
      2,
      1,
      0
    ],
    [
      0,
      2,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      2,
      2,
      2,
      1,
      1,
      2
    ],
    [
      1,
      2,
      2,
      1,
      1,
      2,
      1
    ]
  ],
  "difficulty": "hard" }

get_next_action_func(payload['list'], payload['difficulty'])