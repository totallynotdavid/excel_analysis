pylint $(Get-ChildItem -Recurse -Filter '*.py' | ForEach-Object { $_.FullName })
