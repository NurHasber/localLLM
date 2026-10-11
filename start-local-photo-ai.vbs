' Hidden start: tray icon, no cmd / python window.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonw) Then
  pythonw = root & "\.venv\Scripts\python.exe"
End If
sh.CurrentDirectory = root
sh.Run """" & pythonw & """ -u """ & root & "\tray_app.py""", 0, False
