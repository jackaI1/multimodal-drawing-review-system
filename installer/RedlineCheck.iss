; Inno Setup script for a polished, click-through RedlineCheck installer.
; OPTIONAL / advanced: install Inno Setup (https://jrsoftware.org/isdl.php), open this
; file, and press Build to produce "RedlineCheckSetup.exe". Recipients then run that
; .exe like any normal program (Next -> Next -> Finish). It copies the app, runs the
; same Python/Tesseract/packages setup as the .bat installer, and creates the
; Start-menu and Desktop shortcuts.
;
; Run this from the project ROOT (the folder above "installer"), or adjust SourceDir.

#define AppName "RedlineCheck"
#define AppVer  "1.0"

[Setup]
AppName={#AppName}
AppVersion={#AppVer}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=RedlineCheckSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The setup itself is small; Python/Tesseract are fetched during install.
PrivilegesRequired=admin

[Files]
; Copy the whole project (run the compiler from the project root).
Source: "app.py";              DestDir: "{app}"; Flags: ignoreversion
Source: "comparator.py";       DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt";    DestDir: "{app}"; Flags: ignoreversion
Source: "requirements-lock.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md";           DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE";             DestDir: "{app}"; Flags: ignoreversion
Source: "THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "THIRD_PARTY_LICENSES\*"; DestDir: "{app}\THIRD_PARTY_LICENSES"; Flags: ignoreversion recursesubdirs
Source: "INSTALL.md";          DestDir: "{app}"; Flags: ignoreversion
Source: "redlinecheck\*";      DestDir: "{app}\redlinecheck"; Flags: ignoreversion recursesubdirs
Source: "installer\*";         DestDir: "{app}\installer";    Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\RedlineCheck";       Filename: "{app}\installer\run_redlinecheck.bat";  WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 167
Name: "{group}\Stop RedlineCheck";  Filename: "{app}\installer\stop_redlinecheck.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 131
Name: "{autodesktop}\RedlineCheck"; Filename: "{app}\installer\run_redlinecheck.bat";  WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 167

[Run]
; After copying files, run the (visible) installer to set up Python, Tesseract and the
; Python packages. It shows progress and handles the "Python just installed" re-run.
Filename: "{app}\installer\Install RedlineCheck.bat"; \
  Description: "Set up Python, OCR and components now"; \
  Flags: postinstall shellexec skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
