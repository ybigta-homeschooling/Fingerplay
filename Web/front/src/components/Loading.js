import Loader from "react-loader-spinner"

export function Loading() {
    return (
        <SpinContainer>
            <Loader
                type="Oval"
                color="#3d66ba"
                height={30}
                width={30}
                timeout={3000}           
            />
        </SpinContainer>
    )
}