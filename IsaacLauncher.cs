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
   bool agent=String.Equals(Path.GetFileNameWithoutExtension(Application.ExecutablePath),"RobotAgent",StringComparison.OrdinalIgnoreCase);
   bool learned=File.Exists(Path.Combine(root,"out/ct_appendectomy/agent/policy.pt"));
   string agentPolicy=learned ? "--policy checkpoint --checkpoint out/ct_appendectomy/agent/policy.pt" : "--policy scripted";
   string arguments=agent ? "ct_appendectomy/agent_runner.py "+agentPolicy+" --episodes 0 --out out/ct_appendectomy/agent_desktop" : "ct_appendectomy/run_scene.py --scene out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda --bimanual --frames 360 --interactive --out out/ct_appendectomy/bimanual_desktop";
   var start=new ProcessStartInfo(python,arguments);
   start.WorkingDirectory=root; start.UseShellExecute=false; start.CreateNoWindow=true;
   start.EnvironmentVariables["OMNI_KIT_ACCEPT_EULA"]="YES";
   start.RedirectStandardOutput=true; start.RedirectStandardError=true;
   Directory.CreateDirectory(Path.Combine(root,".cache"));
   using(var log=new StreamWriter(Path.Combine(root,agent ? ".cache/agent-desktop.log" : ".cache/desktop-launch.log"),false)) {
    log.AutoFlush=true; var proc=new Process();proc.StartInfo=start;
    proc.OutputDataReceived+=(o,e)=>{if(e.Data!=null)lock(log){log.WriteLine(e.Data);}};
    proc.ErrorDataReceived+=(o,e)=>{if(e.Data!=null)lock(log){log.WriteLine(e.Data);}};
    proc.Start();proc.BeginOutputReadLine();proc.BeginErrorReadLine();proc.WaitForExit();
   }
  } catch(Exception e) { MessageBox.Show(e.Message,"Isaac launcher"); }
 }
}
