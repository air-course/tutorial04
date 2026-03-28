import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
from cloudpendulumclient.client import Client
import wget
import subprocess
import os
import ast
import time
from misc import download_video
from IPython.display import Video

class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81, inertia=None, torque_limit=np.inf):
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        if inertia is None:
            self.I = mass*length*length
        else:
            self.I = inertia
        self.torque_limit = torque_limit

        self.dof = 1
        self.x = np.zeros(2*self.dof) #position, velocity
        self.t = 0.0 #time

        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee_pos_x = self.l * np.sin(pos)
        ee_pos_y = -self.l*np.cos(pos)
        return [ee_pos_x, ee_pos_y]

    def step(self, tau, dt, integrator="euler"):
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        if integrator == "runge_kutta":
            self.x = self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x = self.euler_integrator(self.t, self.x, dt, tau)
        self.t += dt
        # Store the time series output
        self.t_values.append(self.t)
        self.x_values.append(self.x.copy())
        self.tau_values.append(tau)

    def simulate(self, t0, x0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, x0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        while (self.t <= tf):
            if controller is not None:
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values

    def simulate_and_animate(self, t0, x0, tf, dt, controller=None, integrator="euler", save_video=False):
        """
        simulate and animate the pendulum
        """
        self.set_state(t0, x0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        #fig = plt.figure(figsize=(6,6))
        #self.animation_ax = plt.axes()
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
        ee_plot, = self.animation_ax.plot([], [], "o", markersize=25.0, color="blue")
        bar_plot, = self.animation_ax.plot([], [], "-", lw=5, color="black")
        #text_plot = self.animation_ax.text(0.1, 0.1, [], xycoords="figure fraction")
        self.animation_plots.append(ee_plot)
        self.animation_plots.append(bar_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]
        
        #ps_fig = plt.figure(figsize=(6,6))
        #self.ps_ax = plt.axes()
        #self.ps_plots = []
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        #self.ps_plots.append(ps_plot)
        self.animation_plots.append(ps_plot)

        animation = FuncAnimation(fig, self._animation_step, frames=frames, init_func=self._animation_init, blit=True, repeat=False, interval=dt*1000)
        animation2 = None
        #if phase_plot:
        #    animation2 = FuncAnimation(fig, self._ps_update, init_func=self._ps_init, blit=True, repeat=False, interval=dt*1000)

        if save_video:
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save('pendulum_swingup.mp4', writer=writer)
            #if phase_plot:
            #    Writer2 = mplanimation.writers['ffmpeg']
            #    writer2 = Writer2(fps=60, bitrate=1800)
            #    animation2.save('pendulum_swingup_phase.mp4', writer=writer2)
        #plt.show()
            
        return self.t_values, self.x_values, self.tau_values, animation#, animation2

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_ylim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots:
            ap.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        if controller is not None:
            tau = controller.get_control_output(self.x)
        else:
            tau = 0
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.forward_kinematics(self.x[0])
        self.animation_plots[0].set_data((ee_pos[0],), (ee_pos[1],))
        self.animation_plots[1].set_data([0, ee_pos[0]], [0, ee_pos[1]])

        self._ps_update(0)

        return self.animation_plots

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi, 2*np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.animation_plots:
            ap.set_data([], [])
        return self.animation_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        self.animation_plots[-1].set_data(np.asarray(self.x_values).T[0], np.asarray(self.x_values).T[1])
        return self.animation_plots

    def activate_hardware(self):
        """
        Activate the pendulum hardware
        """    
        import pyCandle

        # Create CANdle object and set FDCAN baudrate to 1Mbps
        self.candle = pyCandle.Candle(pyCandle.CAN_BAUD_8M,True)

        # Ping FDCAN bus in search of drives
        ids = self.candle.ping()

        # Add all found to the update list
        for id in ids:
            self.candle.addMd80(id)

    def run_on_hardware_cloud(self, user_token, x0, tf, dt, controller=None, save_video = True):
        client = Client()
        session_token, livestream_url = client.start_experiment(
            user_token = user_token,
            experiment_type = "SimplePendulum",
            experiment_time = 5.0,
            preparation_time = 5.0,
            initial_state = [x0],
            record = True
        )
        client.set_impedance_controller_params(0,0, session_token)
        meas_time = 0.0
        meas_dt = 0.0
        n = int(tf / dt)
        i = 0
        self.t_values = np.zeros(n)
        self.x_values = np.zeros((2,n))
        self.tau_values = np.zeros(n)
        self.des_tau_values = np.zeros(n)
        meas_dt_vec = np.zeros(n)
        
        while meas_time < tf and i < n:
            start_loop = time.time()
            meas_time += meas_dt
        
            # Measure data
            measured_position = client.get_position(session_token) # Measure position
            measured_velocity = client.get_velocity(session_token) # Measure velocity
            measured_torque = client.get_torque(session_token) # Measure torque

            # controller
            if controller is None:
                tau = 0.0
            else:
                tau = controller.get_control_output(np.array([measured_position, measured_velocity]))
                client.set_torque(tau, session_token)
            
            # add data to matrices
            self.t_values[i] = meas_time
            self.x_values[:,i] = np.array([measured_position, measured_velocity])
            self.tau_values[i] = measured_torque
            self.des_tau_values[i] = tau
                
            self.wait_for_control_loop_end(start_loop, dt)
                
            meas_dt = time.time() - start_loop
            meas_dt_vec[i] = meas_dt
        
            i = i + 1
        
        video_url = client.stop_experiment(session_token)

        if save_video:
            video_path = download_video(video_url)
        else:
            video_path = ""
    
        return self.t_values[:i], self.x_values[:,:i].T, self.tau_values[:i], self.des_tau_values[:i], video_url, video_path

    def wait_for_control_loop_end(self, start_loop, dt):
        """Delay ending a while loop so that it loops at a desired sampling time dt."""
        time_to_pass = start_loop + dt - time.time()
        if time_to_pass <= 0.0:
            return
        
        time.sleep(time_to_pass * 0.7) # sleep
        while time.time() - start_loop < dt: # busy waiting
            pass
        
    def convert_flv_to_mp4(self, input_path, output_path):
        """
        Convert an FLV file to MP4 using FFmpeg.
    
        :param input_path: Path to the input FLV file.
        :param output_path: Path to the output MP4 file.
        """
        command = [
            "ffmpeg",
            "-i", input_path,    # Input file
            "-c:v", "copy",      # Copy video stream
            "-c:a", "copy",      # Copy audio stream
            output_path          # Output file
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode == 0:
            print(f"Conversion successful: {output_path}")
        else:
            print(f"Error during conversion: {process.stderr.decode()}")


    def run_on_hardware_phys(self, tf, dt, controller=None):
        
        import pyCandle
        import time

        # Select pendulum from motor list
        
        # Now we shall loop over all found drives to change control mode and enable them one by one
        for md in self.candle.md80s:
            self.candle.controlMd80SetEncoderZero(md)      #  Reset encoder at current position
            self.candle.controlMd80Mode(md, pyCandle.IMPEDANCE)    # Set mode to impedance control
            self.candle.controlMd80Enable(md, True)     # Enable the drive

        # Begin update loop (it starts in the background)
        self.candle.begin()

        candle_dict = {}
        motornum = 0
        for motor in self.candle.md80s:
            candle_dict[self.candle.md80s[motornum].getId()] = motornum
            motornum += 1

        md80id = 899
        
        md80num = candle_dict[md80id]
        
        # set zero impedance (kp=kd=0) for pure torque control 
        self.candle.md80s[md80num].setImpedanceControllerParams(0, 0)
        
        input("Press bring the pendulum to the starting configuration and press enter to continue...")
    
        tau_scaling = 1.0

        n = int(tf / dt)

        meas_time_vec = np.zeros(n)
        meas_pos = np.zeros(n)
        meas_vel = np.zeros(n)
        meas_tau = np.zeros(n)
        des_tau = np.zeros(n)

        # defining runtime variables
        i = 0
        meas_dt = 0.0
        meas_time = 0.0

        print("Control Loop Started!")
        # Auto update loop is running in the background updating data in candle.md80s vector. Each md80 object can be 
        # Called for data at any time
        while i < n:
            start_loop = time.time()
            meas_time += meas_dt
            
            ## Do your stuff here - START
            
            measured_position = self.candle.md80s[md80num].getPosition()
            measured_velocity = self.candle.md80s[md80num].getVelocity()  
            measured_torque = self.candle.md80s[md80num].getTorque()             
            self.x = np.array([measured_position, measured_velocity])
            
            # Control logic
            if controller is not None:
                tau = controller.get_control_output(self.x)
                tau_scaled = tau*tau_scaling    # physical torque to motor torque
                self.candle.md80s[md80num].setTargetTorque(tau_scaled)
            else:
                tau = 0                
                       
            # Collect data for plotting
            meas_time_vec[i] = meas_time
            meas_pos[i] = measured_position
            meas_vel[i] = measured_velocity    
            meas_tau[i] = self.candle.md80s[md80num].getTorque()/tau_scaling
            des_tau[i] = tau 
                
            ## Do your stuff here - END
            
            i += 1
            exec_time = time.time() - start_loop
            if exec_time > dt:
                print("Control loop is too slow!")
                print("Control frequency:", 1/exec_time, "Hz")
                print("Desired frequency:", 1/dt, "Hz")
                print()
            while time.time() - start_loop < dt:
                pass
            meas_dt = time.time() - start_loop
        print("Control Loop Ended!")

        # Send a few zeros to the motor and then close the update loop
        for i in range(5):
            self.candle.md80s[md80num].setTargetTorque(0.0)
        self.candle.end()
        
        self.t_values = meas_time_vec
        self.x_values = np.vstack((meas_pos, meas_vel)).T
        self.tau_values = meas_tau
        self.des_tau_values = des_tau
        
        return self.t_values, self.x_values, self.tau_values, self.des_tau_values

    def plot_energy(self):
        t = self.t_values
        Kt = np.zeros(len(t))
        Vt = np.zeros(len(t))
        #Et = np.zeros(t.shape[0])
        for i in range(len(t)):
            Kt[i] = 1/2*self.m*(self.l*self.x_values[1][i])**2
            Vt[i] = -self.m*self.g*self.l*np.cos(self.x_values[0][i])

        Et = Kt + Vt
        
        plt.plot(t, Kt, label="Kinetic energy")
        plt.plot(t, Vt, label="Potential energy")
        plt.plot(t, Et, label="Total mechanical energy")
        plt.legend(loc="best")
        plt.show()
            

def plot_timeseries(T, X, U):
    plt.plot(T, np.asarray(X).T[0], label="theta")
    plt.plot(T, np.asarray(X).T[1], label="theta dot")
    plt.plot(T, U, label="u")
    plt.legend(loc="best")
    plt.show()

