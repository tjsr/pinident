def merge_yaml(data1, data2):
  """
  Recursively merge two YAML structures.
  """
  if not isinstance(data1, dict) or not isinstance(data2, dict):
    return data2

  for key, value in data2.items():
    if key in data1:
      data1[key] = merge_yaml(data1[key], value)
    else:
      data1[key] = value

  return data1
