Dim shell, fso, dirPath, rc
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

dirPath = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = dirPath

rc = shell.Run("cmd.exe /c run_gui_hidden.bat", 0, True)

If rc <> 0 Then
  MsgBox "Nao consegui abrir o app (codigo " & rc & ")." & vbCrLf & _
         "Abra a pasta 'logs' e me envie os arquivos:" & vbCrLf & _
         "- logs\\app_stderr.txt" & vbCrLf & _
         "- logs\\launcher_log.txt", 48, "Layout Planner"
End If
