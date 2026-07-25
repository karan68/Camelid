import ExpoModulesCore

public class CamelidRemoteCryptoModule: Module {
  public func definition() -> ModuleDefinition {
    Name("CamelidRemoteCrypto")

    AsyncFunction("setValueAsync") { (value: String) in
    }
  }
}
