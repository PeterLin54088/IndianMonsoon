def UNITEST_streamfunction_global(fname):
    """
    https://derekyuntao.github.io/jekyll-clean-dark/2021/02/mass-stream-func/
    """
    dataset = nc.Dataset(os.path.join(path_ERA5, fname))
    pressure = dataset["plev"][::-1]
    thickness = np.diff(np.insert(pressure, 0, 0))
    lat = dataset["lat"][:]
    v_div = dataset["v"][:, ::-1, :, -1]
    print(np.insert(pressure, 0, 0))
    print(thickness)
    print(v_div.shape)

    #
    tmp = np.insert(v_div, 0, 0, axis=1)
    v_div_interp = (tmp[:, :-1, :] + tmp[:, 1:, :]) / 2
    weighting = 2 * np.pi * 6.371e6 * np.cos(np.deg2rad(lat)) / 9.81
    streamfunction = np.swapaxes(v_div_interp, 1, -1) * thickness
    streamfunction = np.cumsum(streamfunction, axis=-1)
    streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting

    streamfunction_convolution = moving_average(streamfunction, axis=0)

    def animation_generator(sf):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=160)
        ax.set_xticks(np.linspace(-90, 90, 7))
        ax.set_yticks(np.linspace(pressure[0], pressure[-1], 4))

        def init():
            return None

        def run(index):
            ax.cla()
            ax.invert_yaxis()
            tmp = np.mean(sf[index * 5 :: 365], axis=0)
            ax.contourf(
                lat,
                pressure,
                tmp,
                levels=np.linspace(-5e10, 5e10, 8),
                extend="both",
                cmap="RdBu_r",
            )
            ax.contour(
                lat, pressure, tmp, levels=np.linspace(-5e10, 5e10, 8), colors="k"
            )
            ax.set_title(f"day {index*5}")
            return None

        ani = animation.FuncAnimation(
            fig, run, frames=int(365 / 5) + 1, interval=10, init_func=init
        )
        return ani

    # animation_object = animation_generator(streamfunction_convolution)
    # animation_object.save("animation.gif")
    return streamfunction, streamfunction_convolution


def UNITEST_streamfunction_regional(fname):
    """
    https://derekyuntao.github.io/jekyll-clean-dark/2021/02/mass-stream-func/
    """
    dataset = nc.Dataset(os.path.join(path_ERA5, fname))
    pressure = dataset["plev"][::-1]
    thickness = np.diff(np.insert(pressure, 0, 0))
    lon = dataset["lon"][:]
    lat = dataset["lat"][:]
    v_div = dataset["v"][:, ::-1, :, :]
    print(np.insert(pressure, 0, 0))
    print(thickness)
    print(v_div.shape)
    print(lon[0], lon[-1])
    print(lat[0], lat[-1])

    #
    tmp = np.mean(v_div, axis=-1)
    tmp = np.insert(tmp, 0, 0, axis=1)
    v_div_interp = (tmp[:, :-1, :] + tmp[:, 1:, :]) / 2
    weighting = np.deg2rad(lon[-1] - lon[0]) * 6.371e6 * np.cos(np.deg2rad(lat)) / 9.81
    streamfunction = np.swapaxes(v_div_interp, 1, -1) * thickness
    streamfunction = np.cumsum(streamfunction, axis=-1)
    streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting

    streamfunction_convolution = moving_average(streamfunction, axis=0)

    def animation_generator(sf):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 9), dpi=160)
        ax.set_xticks(np.linspace(0, 30, 7))
        ax.set_yticks(np.linspace(pressure[0], pressure[-1], 4))

        def init():
            return None

        def run(index):
            ax.cla()
            ax.invert_yaxis()
            tmp = np.mean(sf[index * 5 :: 365], axis=0)
            ax.contourf(
                lat,
                pressure,
                tmp,
                levels=np.linspace(-5e10, 5e10, 8),
                extend="both",
                cmap="RdBu_r",
            )
            ax.contour(
                lat, pressure, tmp, levels=np.linspace(-5e10, 5e10, 8), colors="k"
            )
            ax.set_title(f"day {index*5}")
            return None

        ani = animation.FuncAnimation(
            fig, run, frames=int(365 / 5) + 1, interval=10, init_func=init
        )
        return ani

    # animation_object = animation_generator(streamfunction_convolution)
    # animation_object.save("animation.gif")
    return streamfunction, streamfunction_convolution

def streamfunction_cases_animation_generator(streamfunction, lat, pressure):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(32, 18), dpi=160, sharex=True, sharey=True
    )
    plt.rcParams.update({"font.size": 28})

    def init():
        return None

    def run(index):
        for (i, j), ax in np.ndenumerate(axes):
            ax.cla()
            match i:
                case 0:
                    ax.invert_yaxis()
                    tmp_year = list(monsoon_onset_sorted.keys())[j]
                    tmp_date = list(monsoon_onset_sorted.values())[j]
                case 1:
                    tmp_year = list(monsoon_onset_sorted.keys())[j - 5]
                    tmp_date = list(monsoon_onset_sorted.values())[j - 5]
            tmp = streamfunction[tmp_year - 1979, index + 100]
            ax.contourf(
                lat,
                pressure,
                tmp,
                levels=np.linspace(-2e10, 2e10, 16),
                extend="both",
                cmap="RdBu_r",
            )
            ax.contour(
                lat, pressure, tmp, levels=np.linspace(-2e10, 2e10, 16), colors="k"
            )

            ax.set_title(f"({tmp_year}, {tmp_date})")
        fig.suptitle(f"Calendar date = {index + 100}")
        return None

    ani = animation.FuncAnimation(fig, run, frames=70, interval=20, init_func=init)
    return ani

def equiv_theta_cases_animation_generator(equiv_theta, lat, pressure):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(32, 18), dpi=160, sharex=True, sharey=True
    )
    plt.rcParams.update({"font.size": 28})

    def init():
        return None

    def run(index):
        for (i, j), ax in np.ndenumerate(axes):
            ax.cla()
            match i:
                case 0:
                    ax.invert_yaxis()
                    tmp_year = list(monsoon_onset_sorted.keys())[j]
                    tmp_date = list(monsoon_onset_sorted.values())[j]
                case 1:
                    tmp_year = list(monsoon_onset_sorted.keys())[j - 5]
                    tmp_date = list(monsoon_onset_sorted.values())[j - 5]
            tmp = equiv_theta[tmp_year - 1979, index + 100]
            ax.contourf(
                lat,
                pressure,
                tmp,
                levels=np.linspace(320, 350, 16),
                extend="both",
                cmap="RdBu_r",
            )
            ax.contour(lat, pressure, tmp, levels=np.linspace(320, 350, 16), colors="k")

            ax.set_title(f"({tmp_year}, {tmp_date})")
        fig.suptitle(f"Calendar date = {index + 100}")
        return None

    ani = animation.FuncAnimation(fig, run, frames=70, interval=20, init_func=init)
    return ani

def equiv_theta_and_streamfunction_composites_animation_generator(
    equiv_theta, streamfunction, equiv_theta_grids, streamfunction_grids
):
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )
    plt.rcParams.update({"font.size": 28})

    def init():
        return None

    def run(index):
        ax.cla()
        ax.invert_yaxis()
        tmp1 = np.mean(
            equiv_theta[
                np.array(list(monsoon_onset_sorted.keys())[:5]) - 1979,
                index + 100,
            ],
            axis=0,
        )
        tmp2 = np.mean(
            streamfunction[
                np.array(list(monsoon_onset_sorted.keys())[:5]) - 1979,
                index + 100,
            ],
            axis=0,
        )
        ax.contourf(
            equiv_theta_grids[1],
            equiv_theta_grids[0],
            tmp1,
            levels=np.linspace(320, 350, 16),
            extend="both",
            cmap="RdBu_r",
        )
        ax.contour(
            streamfunction_grids[1],
            streamfunction_grids[0],
            tmp2,
            levels=np.linspace(-3e10, 3e10, 16),
            colors="k",
        )
        ax.set_xlabel("latitude")
        ax.set_ylabel("pressure (Pa)")
        fig.suptitle(
            f"Contour - streamfunction\nShading - equiv_theta\nCalendar date = {index + 100}"
        )
        plt.tight_layout()
        return None

    ani = animation.FuncAnimation(fig, run, frames=70, interval=100, init_func=init)
    return ani