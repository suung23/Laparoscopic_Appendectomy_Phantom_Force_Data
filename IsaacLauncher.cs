using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
class IsaacLauncher {
 [STAThread] static void Main() {
  try {
   string root=AppDomain.CurrentDomain.BaseDirectory;
   string python=Path.GetFullPath(Path.Combine(root,"../../work/isaac-venv/Scripts/python.exe"));
   if(!File.Exists(python)) throw new FileNotFoundException("Isaac Python environment missing",python);
   var start=new ProcessStartInfo(python,"ct_appendectomy/run_scene.py --retraction-probe --interactive --out out/ct_appendectomy/desktop_probe");
   start.WorkingDirectory=root; start.UseShellExecute=false; start.CreateNoWindow=true;
   start.EnvironmentVariables["OMNI_KIT_ACCEPT_EULA"]="YES";
   start.RedirectStandardOutput=true; start.RedirectStandardError=true;
   Directory.CreateDirectory(Path.Combine(root,".cache"));
   using(var log=new StreamWriter(Path.Combine(root,".cache/desktop-launch.log"),false)) {
    log.AutoFlush=true; var proc=new Process();proc.StartInfo=start;
    proc.OutputDataReceived+=(o,e)=>{if(e.Data!=null)lock(log){log.WriteLine(e.Data);}};
    proc.ErrorDataReceived+=(o,e)=>{if(e.Data!=null)lock(log){log.WriteLine(e.Data);}};
    proc.Start();proc.BeginOutputReadLine();proc.BeginErrorReadLine();proc.WaitForExit();
   }
  } catch(Exception e) { MessageBox.Show(e.Message,"Isaac launcher"); }
 }
}
