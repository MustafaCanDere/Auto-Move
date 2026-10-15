; VisionFlow Studio - Windows kurulum sihirbazı (Inno Setup 6)
; Önce PyInstaller paketi (dist\VisionFlow) oluşturulur; en kolayı: installer\build_windows.bat
; Elle derlemek için:  ISCC.exe /DMyAppVersion=2.3.0 installer\visionflow.iss

#ifndef MyAppVersion
  #define MyAppVersion "2.4.0"
#endif
#define MyAppName "VisionFlow Studio"
#define MyAppExe "VisionFlow.exe"
#define MyProgId "VisionFlow.Akis"
#define RepoRoot ".."

[Setup]
AppId={{C28AECAD-0CD0-48B6-9FEE-261D7A113FEC}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=VisionFlow
DefaultDirName={autopf}\VisionFlow
DefaultGroupName=VisionFlow Studio
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
OutputDir={#RepoRoot}\dist\installer
OutputBaseFilename=VisionFlow-Kurulum-{#MyAppVersion}
SetupIconFile={#RepoRoot}\installer\app.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName}
CloseApplications=yes

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
turkish.FlowFileDesc=VisionFlow akışı
english.FlowFileDesc=VisionFlow akışı (VisionFlow flow)
turkish.FileAssoc=.json dosyalarını VisionFlow ile açılabilir yap ("VisionFlow akışı")
english.FileAssoc=Allow opening .json files with VisionFlow ("VisionFlow akışı")
turkish.AutoStart=Windows ile başlat (seçilen akışı bilgisayar açılınca arayüzsüz çalıştır)
english.AutoStart=Start with Windows (run the chosen flow headless at logon)
turkish.AutoStartGroup=Otomatik çalıştırma:
english.AutoStartGroup=Automatic start:
turkish.FlowPageTitle=Windows ile başlayacak akış
english.FlowPageTitle=Flow to start with Windows
turkish.FlowPageDesc=Bilgisayar açılınca hangi akış çalışsın?
english.FlowPageDesc=Which flow should run when the computer starts?
turkish.FlowPageSub=Akış arayüz olmadan çalışır, çökerse kendiliğinden yeniden başlar. Daha sonra "VisionFlow.exe servis kaldir --ad VisionFlow" ile kapatabilirsiniz.
english.FlowPageSub=The flow runs without the editor and restarts itself after a crash. Remove it later with "VisionFlow.exe servis kaldir --ad VisionFlow".
turkish.FlowPageLabel=Akış dosyası (.json):
english.FlowPageLabel=Flow file (.json):
turkish.FlowFilter=VisionFlow akışı (*.json)|*.json
english.FlowFilter=VisionFlow flow (*.json)|*.json
turkish.FlowMissing=Lütfen var olan bir akış dosyası seçin.
english.FlowMissing=Please choose an existing flow file.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "fileassoc"; Description: "{cm:FileAssoc}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "{cm:AutoStart}"; GroupDescription: "{cm:AutoStartGroup}"; Flags: unchecked

[Files]
; the PyInstaller build (run_visionflow.py + visionflow.spec)
Source: "{#RepoRoot}\dist\VisionFlow\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#RepoRoot}\examples\*"; DestDir: "{app}\examples"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#RepoRoot}\plugins\*"; DestDir: "{app}\plugins"; Flags: onlyifdoesntexist recursesubdirs createallsubdirs uninsneveruninstall
Source: "{#RepoRoot}\models\BURAYA_MODEL_KOYUN.txt"; DestDir: "{app}\models"; Flags: onlyifdoesntexist
Source: "{#RepoRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "{#RepoRoot}\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; flows save recordings and models next to the program: let normal users write there
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\plugins"; Permissions: users-modify

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autoprograms}\{#MyAppName} - Örnekler"; Filename: "{app}\examples"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\{#MyProgId}"; ValueType: string; ValueName: ""; ValueData: "{cm:FlowFileDesc}"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExe},0"; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExe}"" ""%1"""; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\.json\OpenWithProgids"; ValueType: string; ValueName: "{#MyProgId}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExe}\SupportedTypes"; ValueType: string; ValueName: ".json"; ValueData: ""; Flags: uninsdeletekey; Tasks: fileassoc

[Run]
Filename: "{app}\{#MyAppExe}"; Parameters: "servis kur --akis ""{code:GetFlowPath}"" --ad VisionFlow"; StatusMsg: "{cm:AutoStart}"; Tasks: autostart; Flags: runhidden waituntilterminated
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\{#MyAppExe}"; Parameters: "servis kaldir --ad VisionFlow"; Flags: runhidden waituntilterminated; RunOnceId: "VisionFlowServisKaldir"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Code]
var
  FlowPage: TInputFileWizardPage;

procedure InitializeWizard;
begin
  FlowPage := CreateInputFilePage(wpSelectTasks, CustomMessage('FlowPageTitle'),
    CustomMessage('FlowPageDesc'), CustomMessage('FlowPageSub'));
  FlowPage.Add(CustomMessage('FlowPageLabel'), CustomMessage('FlowFilter'), '.json');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = FlowPage.ID) and not WizardIsTaskSelected('autostart');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = FlowPage.ID) and not FileExists(FlowPage.Values[0]) then
  begin
    MsgBox(CustomMessage('FlowMissing'), mbError, MB_OK);
    Result := False;
  end;
end;

function GetFlowPath(Param: String): String;
begin
  Result := FlowPage.Values[0];
end;
