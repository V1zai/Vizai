import React, { useState } from "react";
import {
  SafeAreaView,
  View,
  ScrollView,
  ImageBackground,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
} from "react-native";

export default function DetectionToggle() {
  const [isDetecting, setIsDetecting] = useState(false);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={{ flexGrow: 1 }}>
        <View style={styles.view}>
          <View style={styles.column}>
            <View style={styles.box} />

            <TouchableOpacity
              onPress={() => setIsDetecting(!isDetecting)}
              activeOpacity={0.8}
            >
              <ImageBackground
                source={{
                  uri: isDetecting
                    ? "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/rtoi4e38_expires_30_days.png"
                    : "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/r4dunbsh_expires_30_days.png",
                }}
                resizeMode="stretch"
                style={styles.view2}
              >
                <View style={styles.view3}>
                  <Text style={styles.text}>
                    {isDetecting ? "Stop Detection" : "Start Detection"}
                  </Text>
                </View>
              </ImageBackground>
            </TouchableOpacity>

            <View style={styles.row}>
              <Image
                source={{
                  uri: isDetecting
                    ? "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/9tsxj777_expires_30_days.png"
                    : "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/ob7fmp1s_expires_30_days.png",
                }}
                resizeMode="stretch"
                style={styles.image}
              />
              <Image
                source={{
                  uri: isDetecting
                    ? "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/7770hqcp_expires_30_days.png"
                    : "https://storage.googleapis.com/tagjs-prod.appspot.com/v1/bxKexi88Se/yk90osu0_expires_30_days.png",
                }}
                resizeMode="stretch"
                style={styles.image2}
              />
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  box: {
    height: 575,
    alignSelf: "stretch",
    backgroundColor: "#D9D9D9",
    borderColor: "#D9D9D9",
    borderRadius: 32,
    borderWidth: 1,
    marginBottom: 61,
    shadowColor: "#00000040",
    shadowOpacity: 0.3,
    shadowOffset: {
      width: 1,
      height: 6,
    },
    shadowRadius: 4,
    elevation: 4,
  },
  column: {
    alignItems: "center",
    backgroundColor: "#67A8BA",
    borderColor: "#1E1E1E",
    borderRadius: 24,
    borderWidth: 1,
    padding: 15,
    marginHorizontal: 12,
  },
  image: {
    width: 14,
    height: 14,
    marginRight: 9,
  },
  image2: {
    width: 14,
    height: 14,
  },
  row: {
    flexDirection: "row",
  },
  scrollView: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  text: {
    color: "#FFFFFF",
    fontSize: 32,
    marginHorizontal: 39,
  },
  view: {
    backgroundColor: "#1E1E1E",
    paddingVertical: 13,
  },
  view2: {
    alignSelf: "stretch",
    paddingVertical: 9,
    marginBottom: 72,
    marginHorizontal: 5,
  },
  view3: {
    backgroundColor: "#579DB0",
    paddingVertical: 10,
    marginHorizontal: 10,
  },
});
